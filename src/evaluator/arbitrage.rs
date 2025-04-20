use crate::evaluator::{MevStrategy, MevOpportunity, MevStrategyEvaluator, RiskLevel};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use tracing::{debug, info, trace, warn};
use std::collections::HashMap;
use anyhow::anyhow;
use super::utils; // Import the new utils module
use crate::market_data::MarketData; // Import MarketData
use crate::types::{DecodedInstructionInfo, DecodedTransactionDetails}; // Path should be correct now
use solana_client::rpc_client::RpcClient;
use std::sync::Arc;
use crate::executor::TransactionExecutor;
use crate::market_data::MarketDataCollector;
use crate::config::Settings;

/// Configuration for arbitrage opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageConfig {
    /// Minimum percentage profit required (e.g., 0.005 = 0.5%)
    pub min_profit_percentage: f64,
    /// Minimum USD value for trade to be considered
    pub min_trade_value_usd: f64,
    /// Maximum price impact allowed for arbitrage trades
    pub max_price_impact_percent: f64,
    /// Maximum time window for arbitrage to be valid (milliseconds)
    pub max_time_window_ms: u64,
    /// Maximum percentage of pool liquidity to use
    pub max_pool_utilization_percent: f64,
    /// Gas price in lamports (for profit calculation)
    pub gas_price_lamports: u64,
    /// Average instruction cost in compute units
    pub avg_instruction_cu: u64,
    /// Priority fee in microlamports per CU (post fee-markets)
    pub priority_fee_microlamports: u64,
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            min_profit_percentage: 0.005, // 0.5%
            min_trade_value_usd: 100.0,  // $100 minimum
            max_price_impact_percent: 1.0, // 1% max price impact
            max_time_window_ms: 2000,    // 2 second window
            max_pool_utilization_percent: 10.0, // Use at most 10% of pool liquidity
            gas_price_lamports: 20000,   // Higher gas price estimate post fee-markets
            avg_instruction_cu: 200000,  // Average instruction cost in compute units
            priority_fee_microlamports: 1000, // Priority fee (1000 microlamports per CU)
        }
    }
}

/// Metadata for arbitrage opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageMetadata {
    /// DEX where the first trade occurs
    pub source_dex: String,
    /// DEX where the second trade occurs
    pub target_dex: String,
    /// Token path for the arbitrage (A -> B -> A)
    pub token_path: Vec<String>,
    /// Estimated price difference (percentage)
    pub price_difference_percent: f64,
    /// Estimated gas costs in USD
    pub estimated_gas_cost_usd: f64,
    /// Optimal trade size in USD
    pub optimal_trade_size_usd: f64,
    /// Price impact of the trade
    pub price_impact_percent: f64,
    /// Pool liquidity for source pool in USD
    pub source_liquidity_usd: f64,
    /// Pool liquidity for target pool in USD
    pub target_liquidity_usd: f64,
    /// Estimated number of instructions needed
    pub estimated_instruction_count: u64,
    /// Estimated compute units needed
    pub estimated_compute_units: u64,
    /// Timestamp when the opportunity was identified
    pub timestamp: i64,
}

/// Evaluator for arbitrage opportunities
pub struct ArbitrageEvaluator {
    config: ArbitrageConfig,
    rpc_client: Arc<RpcClient>,
}

impl ArbitrageEvaluator {
    /// Create a new arbitrage evaluator with default configuration
    pub fn new(rpc_client: Arc<RpcClient>) -> Self {
        Self {
            config: ArbitrageConfig::default(),
            rpc_client,
        }
    }
    
    /// Create a new arbitrage evaluator with custom configuration
    pub fn with_config(config: ArbitrageConfig, rpc_client: Arc<RpcClient>) -> Self {
        Self { config, rpc_client }
    }
    
    /// Calculate the potential profit from an arbitrage opportunity
    fn calculate_profit(&self, source_price: f64, target_price: f64, trade_size_usd: f64, gas_cost_usd: f64) -> f64 {
        let profit_percentage = (target_price - source_price) / source_price;
        (trade_size_usd * profit_percentage) - gas_cost_usd
    }
    
    /// Calculate optimal trade size based on pool liquidity and price impact
    fn calculate_optimal_size(&self, source_liquidity: f64, target_liquidity: f64, price_diff_percent: f64) -> f64 {
        // Note: Liquidity values are now expected to be real (or 0.0 if unavailable)
        // Remove placeholder warnings/floors
        // warn!(target: "arbitrage_evaluator", "Using PLACEHOLDER liquidity for optimal size calculation.");
        // let source_liquidity = source_liquidity.max(10000.0); // Placeholder floor removed
        // let target_liquidity = target_liquidity.max(10000.0); // Placeholder floor removed
        
        if source_liquidity <= 0.0 || target_liquidity <= 0.0 {
            warn!(target: "arbitrage_evaluator", source_liq=%source_liquidity, target_liq=%target_liquidity, "Pool liquidity is zero or negative, using minimum trade size");
            return self.config.min_trade_value_usd;
        }
        
        // Start with a percentage based on price difference
        // Larger price differences can use larger positions
        let base_percentage: f64 = if price_diff_percent < 0.5 {
            0.5 // Very small price difference, be conservative
        } else if price_diff_percent < 1.0 {
            1.0 // Small price difference
        } else if price_diff_percent < 2.0 {
            2.0 // Medium price difference
        } else if price_diff_percent < 5.0 {
            3.0 // Large price difference
        } else {
            5.0 // Very large price difference
        };
        
        // Cap at configured maximum
        let percentage = base_percentage.min(self.config.max_pool_utilization_percent);
        
        // Calculate size based on percentage of liquidity
        let size = (source_liquidity + target_liquidity) * (percentage / 200.0);
        
        // Ensure minimum trade size
        size.max(self.config.min_trade_value_usd)
    }
    
    /// Estimate gas cost for arbitrage transaction
    fn estimate_gas_cost(&self, instruction_count: u64, sol_price_usd: f64) -> f64 {
        // Note: sol_price_usd is now expected to be real (or a default if fetch failed)
        // Remove placeholder warnings/floors
        // if sol_price_usd <= 0.0 {
        //     warn!(target: "arbitrage_evaluator", "Using PLACEHOLDER SOL price for gas calculation.");
        // }
        // let sol_price_usd = sol_price_usd.max(100.0); // Placeholder floor removed
        
        // Estimate total compute units based on instruction count
        let compute_units = instruction_count * self.config.avg_instruction_cu;
        
        // Base gas cost in lamports
        let base_gas_cost_lamports = compute_units * self.config.gas_price_lamports / 100_000;
        
        // Priority fee in lamports
        let priority_fee_lamports = compute_units * self.config.priority_fee_microlamports / 1_000_000;
        
        // Total gas cost in lamports
        let total_gas_cost_lamports = base_gas_cost_lamports + priority_fee_lamports;
        
        // Convert to SOL
        let gas_cost_sol = total_gas_cost_lamports as f64 / 1_000_000_000.0;
        
        let gas_cost_usd = gas_cost_sol * sol_price_usd;
        
        debug!(target: "arbitrage_evaluator", 
               instruction_count = instruction_count,
               compute_units = compute_units,
               base_cost_lamports = base_gas_cost_lamports,
               priority_fee_lamports = priority_fee_lamports,
               gas_cost_sol = gas_cost_sol,
               gas_cost_usd = gas_cost_usd,
               "Estimated gas cost");
               
        gas_cost_usd
    }
    
    /// Determine risk level based on the arbitrage opportunity
    fn determine_risk_level(&self, price_diff_percent: f64, liquidity_usd: f64, price_impact: f64) -> RiskLevel {
        // Note: liquidity_usd is now expected to be real (or 0.0)
        // Remove placeholder warnings/floors
        // warn!(target: "arbitrage_evaluator", "Using PLACEHOLDER liquidity for risk level determination.");
        // let liquidity_usd = liquidity_usd.max(10000.0); // Placeholder floor removed
        
        // Extremely high price differences (> 5%) are suspicious and potentially risky
        if price_diff_percent > 5.0 {
            return RiskLevel::High;
        }
        
        // Low liquidity or high price impact indicate higher risk
        if liquidity_usd < 10000.0 || price_impact > 1.0 {
            return RiskLevel::High;
        } else if liquidity_usd < 50000.0 || price_impact > 0.5 {
            return RiskLevel::Medium;
        } else {
            return RiskLevel::Low;
        }
    }
}

#[async_trait]
impl MevStrategyEvaluator for ArbitrageEvaluator {
    fn strategy_type(&self) -> MevStrategy {
        MevStrategy::Arbitrage
    }
    
    async fn evaluate(&self, data: &Value) -> Result<Option<MevOpportunity>> {
        trace!(target: "arbitrage_evaluator", "Evaluating potential arbitrage opportunity");
        
        // --- Parse Context Passed from OpportunityEvaluator --- 
        let market_context = data.get("market_context").ok_or_else(|| {
            warn!(target: "arbitrage_evaluator", "Missing 'market_context' in input data");
            anyhow!("Missing 'market_context' in input data")
        })?;
        let decoded_context = data.get("decoded_details"); // Optional

        // Extract SOL price (use default if fetch failed)
        let sol_usd_price = market_context
            .get("sol_usd_price")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| {
                 warn!(target: "arbitrage_evaluator", "Missing or invalid SOL price in context, using default.");
                 150.0 // Default SOL price
            });

        // Extract pair market data
        let pair_market_data: Option<MarketData> = market_context
            .get("pair_market_data")
            .and_then(|v| serde_json::from_value(v.clone()).ok()); // Deserialize Option<MarketData>

        // Extract liquidity (use 0.0 if MarketData is missing or liquidity is placeholder -1.0)
        let pool_liquidity = pair_market_data
            .as_ref()
            .map_or(0.0, |md| if md.liquidity == -1.0 { 0.0 } else { md.liquidity });
            
        // --- Extract Details from Decoded Context (If Available) --- 
        let mut parsed_token_pair: Option<(String, String)> = None;
        let mut parsed_input_amount: Option<u64> = None;
        let mut parsed_min_output_amount: Option<u64> = None;
        
        if let Some(decoded_json) = decoded_context {
            if !decoded_json.is_null() { // Check if it's not null before trying to parse
                 match serde_json::from_value::<DecodedTransactionDetails>(decoded_json.clone()) {
                    Ok(details) => {
                        if let Some(primary_ix) = details.primary_instruction {
                            // Prefer decoded details if available
                            if let (Some(input_m), Some(output_m)) = (primary_ix.input_mint, primary_ix.output_mint) {
                                parsed_token_pair = Some((input_m, output_m));
                            }
                            parsed_input_amount = primary_ix.input_amount;
                            parsed_min_output_amount = primary_ix.minimum_output_amount;
                             debug!(target: "arbitrage_evaluator", "Parsed details from decoded context");
                        }
                    },
                    Err(e) => {
                        warn!(target: "arbitrage_evaluator", error=%e, "Failed to deserialize DecodedTransactionDetails from context");
                    }
                }
            }
        }
        
        // Use parsed pair or fallback to placeholder
        let token_pair = parsed_token_pair.unwrap_or_else(|| {
            warn!(target: "arbitrage_evaluator", "Using PLACEHOLDER token pair (decoding failed or unavailable)");
            ("SOL".to_string(), "USDC".to_string())
        });

        // --- Parse Price Data (Legacy - Remove when context passing is complete) --- 
        let real_dex_prices_value = data.get("real_dex_prices").ok_or_else(|| {
            warn!(target: "arbitrage_evaluator", "Missing 'real_dex_prices' in input data (legacy)");
            anyhow!("Missing 'real_dex_prices' in input data (legacy)")
        })?;
        let dex_prices_opt: HashMap<String, Option<f64>> = serde_json::from_value(real_dex_prices_value.clone())?;
        let dex_prices: HashMap<String, f64> = dex_prices_opt
            .into_iter()
            .filter_map(|(k, v)| v.map(|price| (k, price)))
            .collect();
        // --- End Parsing --- 
        
        // Remove internal placeholders
        // let sol_price_usd = 150.0; // Placeholder removed
        // let placeholder_liquidity = 100000.0; // Placeholder removed
        // warn!(target: "arbitrage_evaluator", "Using PLACEHOLDER token pair, SOL price, and liquidity values."); // Warning removed

        if dex_prices.len() < 2 {
            debug!(target: "arbitrage_evaluator", "Insufficient valid DEX price data (need at least 2)");
            return Ok(None);
        }
        
        // Sort DEXes by price to find best buy and sell opportunities
        let mut dex_price_pairs: Vec<(&String, &f64)> = dex_prices.iter().collect();
        dex_price_pairs.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Lowest price is best for buying (source)
        let source_dex = dex_price_pairs.first().unwrap().0.clone();
        let source_price = *dex_price_pairs.first().unwrap().1;
        
        // Highest price is best for selling (target)
        let target_dex = dex_price_pairs.last().unwrap().0.clone();
        let target_price = *dex_price_pairs.last().unwrap().1;
        
        // Calculate price difference percentage
        let price_diff_percentage = (target_price - source_price) / source_price * 100.0;
        
        trace!(target: "arbitrage_evaluator", 
               source_dex = %source_dex,
               target_dex = %target_dex,
               source_price = source_price,
               target_price = target_price,
               price_diff_percentage = price_diff_percentage,
               "Found potential arbitrage between DEXes");
        
        // Check if price difference meets minimum threshold
        if price_diff_percentage < self.config.min_profit_percentage * 100.0 {
            trace!(target: "arbitrage_evaluator", 
                   price_diff = price_diff_percentage,
                   min_required = self.config.min_profit_percentage * 100.0,
                   "Price difference below minimum threshold");
            return Ok(None);
        }
        
        // Use the parsed (potentially 0.0) liquidity data 
        let source_liquidity = pool_liquidity; 
        let target_liquidity = pool_liquidity; 
        
        // Calculate optimal trade size using potentially real liquidity
        let optimal_trade_size = self.calculate_optimal_size(
            source_liquidity, 
            target_liquidity, 
            price_diff_percentage
        );
        // For arbitrage, we use the calculated optimal size
        let trade_size_usd = optimal_trade_size; 

        // Calculate price impact using calculated trade size and potentially real liquidity
        let source_price_impact = utils::calculate_price_impact(trade_size_usd, source_liquidity);
        let target_price_impact = utils::calculate_price_impact(trade_size_usd, target_liquidity);
        let total_price_impact = source_price_impact + target_price_impact;
        
        // Check if price impact is acceptable
        if total_price_impact > self.config.max_price_impact_percent {
             debug!(target: "arbitrage_evaluator", impact=total_price_impact, size=trade_size_usd, "Price impact too high for calculated optimal size");
            return Ok(None);
        }
        
        // Estimate gas costs using parsed SOL price
        let estimated_instruction_count = 8; 
        let estimated_gas_cost = self.estimate_gas_cost(estimated_instruction_count, sol_usd_price);
        
        // Calculate Profit after impact (approximation)
        // Adjust price diff by impact: effective_diff = diff% - impact%
        let effective_price_diff_percentage = price_diff_percentage - total_price_impact;
        let gross_profit = trade_size_usd * (effective_price_diff_percentage / 100.0);
        let net_profit = gross_profit - estimated_gas_cost;
        
        // Final Profitability Check
        if net_profit <= 0.0 {
             debug!(target: "arbitrage_evaluator", profit=net_profit, size=trade_size_usd, "Not profitable after impact and gas");
            return Ok(None);
        }
        
        // Determine risk level using potentially real liquidity
        let risk_level = self.determine_risk_level(
            price_diff_percentage, 
            source_liquidity.min(target_liquidity),
            total_price_impact
        );
        
        // Create metadata using calculated trade size and parsed pair
        let metadata = ArbitrageMetadata {
            source_dex, 
            target_dex, 
            token_path: vec![token_pair.0.clone(), token_pair.1.clone(), token_pair.0.clone()], 
            price_difference_percent: price_diff_percentage,
            estimated_gas_cost_usd: estimated_gas_cost, 
            optimal_trade_size_usd: trade_size_usd, // Report the calculated optimal size
            price_impact_percent: total_price_impact, 
            source_liquidity_usd: source_liquidity, 
            target_liquidity_usd: target_liquidity, 
            estimated_instruction_count,
            estimated_compute_units: estimated_instruction_count * self.config.avg_instruction_cu,
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        // Log the opportunity
        info!(target: "arbitrage_evaluator", 
              tokens = ?metadata.token_path,
              source = %metadata.source_dex,
              target = %metadata.target_dex,
              price_diff = %metadata.price_difference_percent,
              estimated_profit = net_profit,
              "Arbitrage opportunity found");
        
        // Create the opportunity using calculated trade size
        let opportunity = MevOpportunity {
            strategy: MevStrategy::Arbitrage,
            estimated_profit: net_profit,
            confidence: 0.8,
            risk_level,
            required_capital: trade_size_usd, // Use calculated optimal size
            execution_time: 500,
            involved_tokens: vec![token_pair.0.clone(), token_pair.1.clone()], 
            allowed_output_tokens: vec![token_pair.0.clone(), token_pair.1.clone()], 
            allowed_programs: vec![],
            max_instructions: 10,
            metadata: serde_json::to_value(metadata)?,
            score: None,
            decision: None,
        };
        
        Ok(Some(opportunity))
    }

    async fn validate(&self, opportunity: &MevOpportunity) -> Result<bool> {
        // Extract metadata
        let metadata: ArbitrageMetadata = match serde_json::from_value(opportunity.metadata.clone()) {
            Ok(meta) => meta,
            Err(e) => {
                warn!(target: "arbitrage_evaluator", error = %e, "Failed to deserialize opportunity metadata");
                return Ok(false);
            }
        };
        
        // Check if opportunity is still profitable
        if opportunity.estimated_profit <= 0.0 {
            debug!(target: "arbitrage_evaluator", "Opportunity no longer profitable");
            return Ok(false);
        }
        
        // Check if opportunity is too old (based on max time window)
        let age = chrono::Utc::now()
            .signed_duration_since(
                chrono::DateTime::<chrono::Utc>::from_timestamp(metadata.timestamp, 0)
                    .unwrap_or_else(|| chrono::Utc::now())
            )
            .num_milliseconds();
            
        if age as u64 > self.config.max_time_window_ms {
            debug!(target: "arbitrage_evaluator", 
                   age_ms = age,
                   max_window_ms = self.config.max_time_window_ms,
                   "Opportunity too old");
            return Ok(false);
        }
        
        // Additional validation logic would go here
        // For example, recheck current prices to verify the arbitrage still exists
        
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_arbitrage_evaluation() {
        let evaluator = ArbitrageEvaluator::new();
        
        // Create sample data
        let data = serde_json::json!({
            "transaction": {
                "signature": "test_signature",
                "logs": ["Program TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA invoke", 
                         "Transfer 100 from So11111111111111111111111111111111111111112 to EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"]
            },
            "dex_prices": {
                "Jupiter": 1.0,
                "Raydium": 1.02,
                "Orca": 0.99
            },
            "dex_pools": {
                "Jupiter": {
                    "liquidity_usd": 1000000.0,
                    "price": 1.0
                },
                "Raydium": {
                    "liquidity_usd": 800000.0,
                    "price": 1.02
                },
                "Orca": {
                    "liquidity_usd": 700000.0,
                    "price": 0.99
                }
            }
        });
        
        // Evaluate the opportunity
        let result = evaluator.evaluate(&data).await.unwrap();
        
        // Should find an opportunity
        assert!(result.is_some());
        
        if let Some(opportunity) = result {
            // Verify the opportunity details
            assert_eq!(opportunity.strategy, MevStrategy::Arbitrage);
            assert!(opportunity.estimated_profit > 0.0);
            
            // Extract metadata
            let metadata: ArbitrageMetadata = serde_json::from_value(opportunity.metadata).unwrap();
            
            // Check source and target DEXes
            assert_eq!(metadata.source_dex, "Orca");  // Lowest price
            assert_eq!(metadata.target_dex, "Raydium"); // Highest price
            
            // Check price difference
            assert!(metadata.price_difference_percent > 0.0);
            
            // Check tokens involved
            assert_eq!(metadata.token_path.len(), 3);
            assert_eq!(metadata.token_path[0], metadata.token_path[2]); // A->B->A
        }
    }
    
    #[tokio::test]
    async fn test_price_impact_calculation() {
        let evaluator = ArbitrageEvaluator::new();
        
        // Test various scenarios
        let impact1 = evaluator.calculate_price_impact(10000.0, 1000000.0); // 1% of liquidity
        let impact2 = evaluator.calculate_price_impact(100000.0, 1000000.0); // 10% of liquidity
        
        // Check results - impact should increase with trade size
        assert!(impact1 < impact2);
        
        // With 1% of liquidity, impact should be roughly 1%
        assert!(impact1 > 0.9 && impact1 < 1.1);
    }
    
    #[tokio::test]
    async fn test_optimal_size_calculation() {
        let evaluator = ArbitrageEvaluator::new();
        
        // Test with different price differences and liquidity
        let size1 = evaluator.calculate_optimal_size(1000000.0, 2000000.0, 0.3); // Small price diff
        let size2 = evaluator.calculate_optimal_size(1000000.0, 2000000.0, 3.0); // Large price diff
        let size3 = evaluator.calculate_optimal_size(100000.0, 2000000.0, 3.0);  // Limited liquidity
        
        // Check results - size should scale with price difference
        assert!(size1 < size2);
        
        // Size should be limited by smallest pool
        assert!(size3 < size2);
        
        // Size should not exceed pool utilization percentage
        assert!(size2 <= 1000000.0 * evaluator.config.max_pool_utilization_percent / 100.0);
    }
} 