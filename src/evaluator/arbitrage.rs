use crate::evaluator::{MevStrategy, MevOpportunity, MevStrategyEvaluator, RiskLevel};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use tracing::{debug, info, trace, warn};
use std::collections::HashMap;
use anyhow::anyhow;

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
}

impl ArbitrageEvaluator {
    /// Create a new arbitrage evaluator with default configuration
    pub fn new() -> Self {
        Self {
            config: ArbitrageConfig::default(),
        }
    }
    
    /// Create a new arbitrage evaluator with custom configuration
    pub fn with_config(config: ArbitrageConfig) -> Self {
        Self { config }
    }
    
    /// Extract token pairs from transaction data
    fn extract_token_pairs(&self, data: &Value) -> Option<(String, String)> {
        // Define common token patterns at the function level for use in all blocks
        let token_patterns = [
            ("SOL", "So11111111111111111111111111111111111111112"),
            ("USDC", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
            ("USDT", "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"),
            ("ETH", "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"),
            ("BTC", "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E"),
            ("JUP", "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"),
            ("BONK", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"),
        ];

        // First, try to extract directly from JSON
        if let Some(token1) = data.get("token1").and_then(|t| t.as_str()) {
            if let Some(token2) = data.get("token2").and_then(|t| t.as_str()) {
                trace!(target: "arbitrage_evaluator", "Found token pair in JSON: {}/{}", token1, token2);
                return Some((token1.to_string(), token2.to_string()));
            }
        }
        
        // Extract transaction logs and look for token transfers
        if let Some(transaction) = data.get("transaction") {
            if let Some(logs) = transaction.get("logs") {
                trace!(target: "arbitrage_evaluator", "Analyzing logs for token transfers");
                
                // Look for token transfer logs
                if let Some(logs_array) = logs.as_array() {
                    let mut found_tokens = Vec::new();
                    
                    for log in logs_array {
                        if let Some(log_str) = log.as_str() {
                            // Look for token transfers in logs
                            if log_str.contains("spl-token") && log_str.contains("Transfer") {
                                // Check for known token mints in the log
                                for (symbol, mint) in &token_patterns {
                                    if log_str.contains(mint) {
                                        found_tokens.push(symbol.to_string());
                                    }
                                }
                            }
                        }
                    }
                    
                    // If we found at least two different tokens, return them as a pair
                    if found_tokens.len() >= 2 {
                        // Remove duplicates and ensure we have a distinct pair
                        found_tokens.sort();
                        found_tokens.dedup();
                        
                        if found_tokens.len() >= 2 {
                            return Some((found_tokens[0].clone(), found_tokens[1].clone()));
                        }
                    }
                }
            }
        }
        
        // If we still don't have a pair, try to extract from program IDs and account keys
        if let Some(transaction) = data.get("transaction") {
            if let Some(message) = transaction.get("message") {
                if let Some(account_keys) = message.get("accountKeys") {
                    if let Some(keys_array) = account_keys.as_array() {
                        // Check for Jupiter, Raydium, or Orca program IDs
                        let dex_program_found = keys_array.iter().any(|key| {
                            key.as_str().map_or(false, |key_str| {
                                key_str == "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4" ||
                                key_str == "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8" ||
                                key_str == "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
                            })
                        });
                        
                        if dex_program_found {
                            // Check for token mints
                            let mut token_mints = Vec::new();
                            for key in keys_array {
                                if let Some(key_str) = key.as_str() {
                                    for (symbol, mint) in &token_patterns {
                                        if key_str == *mint {
                                            token_mints.push(symbol.to_string());
                                        }
                                    }
                                }
                            }
                            
                            if token_mints.len() >= 2 {
                                token_mints.sort();
                                token_mints.dedup();
                                if token_mints.len() >= 2 {
                                    return Some((token_mints[0].clone(), token_mints[1].clone()));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // If we still don't have a pair from transaction data, use default pair only as last resort
        warn!(target: "arbitrage_evaluator", "Could not extract token pair from transaction data, using default SOL/USDC pair");
        Some(("SOL".to_string(), "USDC".to_string()))
    }
    
    /// Calculate the potential profit from an arbitrage opportunity
    fn calculate_profit(&self, source_price: f64, target_price: f64, trade_size_usd: f64, gas_cost_usd: f64) -> f64 {
        let profit_percentage = (target_price - source_price) / source_price;
        (trade_size_usd * profit_percentage) - gas_cost_usd
    }
    
    /// Calculate optimal trade size based on pool liquidity and price impact
    fn calculate_optimal_size(&self, source_liquidity: f64, target_liquidity: f64, price_diff_percent: f64) -> f64 {
        // Determine the limiting pool (smaller of the two)
        let min_liquidity = source_liquidity.min(target_liquidity);
        
        if min_liquidity <= 0.0 {
            warn!(target: "arbitrage_evaluator", "Pool liquidity is zero or negative, using minimum trade size");
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
        let size = min_liquidity * (percentage / 100.0);
        
        // Ensure minimum trade size
        size.max(self.config.min_trade_value_usd)
    }
    
    /// Calculate price impact based on trade size and pool liquidity
    fn calculate_price_impact(&self, trade_size_usd: f64, pool_liquidity_usd: f64) -> f64 {
        if pool_liquidity_usd <= 0.0 {
            warn!(target: "arbitrage_evaluator", "Pool liquidity is zero or negative, assuming high price impact");
            return 5.0; // Assume 5% impact as a conservative estimate
        }
        
        // Using constant product formula (x * y = k) to calculate price impact
        // For a trade consuming trade_size_usd of a pool with pool_liquidity_usd
        let trade_ratio = trade_size_usd / pool_liquidity_usd;
        
        // Using a more accurate price impact formula: 
        // impact = trade_ratio / (2 - trade_ratio) * 100 for x*y=k pools
        let impact = trade_ratio / (2.0 - trade_ratio.min(1.0)) * 100.0;
        
        debug!(target: "arbitrage_evaluator", 
               trade_size = trade_size_usd, 
               pool_liquidity = pool_liquidity_usd,
               trade_ratio = trade_ratio,
               impact = impact,
               "Calculated price impact");
               
        impact
    }
    
    /// Estimate gas cost for arbitrage transaction
    fn estimate_gas_cost(&self, instruction_count: u64) -> f64 {
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
        
        // Get SOL price from data or use reasonable default
        let sol_price_usd = 100.0; // Should be replaced with actual price from market data
        
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
    
    /// Extract DEX prices from data provided by OpportunityEvaluator
    fn extract_dex_prices(&self, data: &Value, token_pair: &(String, String)) -> Option<HashMap<String, f64>> {
        // Check if prices are already available in the data
        if let Some(prices) = data.get("dex_prices") {
            if let Some(prices_obj) = prices.as_object() {
                let mut dex_prices = HashMap::new();
                
                for (dex, price_data) in prices_obj {
                    if let Some(price_value) = price_data.as_f64() {
                        dex_prices.insert(dex.clone(), price_value);
                    }
                }
                
                if !dex_prices.is_empty() {
                    return Some(dex_prices);
                }
            }
        }
        
        // If we have pool liquidity data, we can extract prices
        if let Some(pools) = data.get("dex_pools") {
            if let Some(pools_obj) = pools.as_object() {
                let mut dex_prices = HashMap::new();
                
                for (dex, pool_data) in pools_obj {
                    if let Some(pool_obj) = pool_data.as_object() {
                        if let Some(price) = pool_obj.get("price").and_then(|p| p.as_f64()) {
                            dex_prices.insert(dex.clone(), price);
                        }
                    }
                }
                
                if !dex_prices.is_empty() {
                    return Some(dex_prices);
                }
            }
        }
        
        warn!(target: "arbitrage_evaluator", "No DEX prices found in data, using simulation data");
        
        // For simulation/testing, create some sample prices
        // This should be replaced with actual price data in production
        let mut simulated_prices = HashMap::new();
        simulated_prices.insert("Jupiter".to_string(), 1.0);
        simulated_prices.insert("Raydium".to_string(), 1.005);
        simulated_prices.insert("Orca".to_string(), 0.995);
        
        Some(simulated_prices)
    }
    
    /// Extract pool liquidity from data provided by OpportunityEvaluator
    fn extract_pool_liquidity(&self, data: &Value, dex: &str) -> f64 {
        // Check if liquidity data is available in the data
        if let Some(pools) = data.get("dex_pools") {
            if let Some(pools_obj) = pools.as_object() {
                if let Some(pool_data) = pools_obj.get(dex) {
                    if let Some(pool_obj) = pool_data.as_object() {
                        if let Some(liquidity) = pool_obj.get("liquidity_usd").and_then(|l| l.as_f64()) {
                            return liquidity;
                        }
                    }
                }
            }
        }
        
        // If not found, return a reasonable default based on the DEX
        match dex {
            "Jupiter" => 500000.0,  // $500k default for Jupiter
            "Raydium" => 300000.0,  // $300k default for Raydium
            "Orca" => 200000.0,     // $200k default for Orca
            _ => 100000.0,          // $100k default for unknown DEXes
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
        
        // Extract token pair from the transaction or data
        let token_pair = match self.extract_token_pairs(data) {
            Some(pair) => pair,
            None => {
                trace!(target: "arbitrage_evaluator", "No token pair found in transaction");
                return Ok(None);
            }
        };
        
        debug!(target: "arbitrage_evaluator", 
               token_1 = %token_pair.0, 
               token_2 = %token_pair.1, 
               "Extracted token pair for arbitrage evaluation");
        
        // Extract DEX prices from data
        let dex_prices = match self.extract_dex_prices(data, &token_pair) {
            Some(prices) => prices,
            None => {
                debug!(target: "arbitrage_evaluator", "No DEX prices available for evaluation");
                return Ok(None);
            }
        };
        
        if dex_prices.len() < 2 {
            debug!(target: "arbitrage_evaluator", "Insufficient DEX price data for arbitrage (need at least 2 DEXes)");
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
        
        // Get pool liquidity data
        let source_liquidity = self.extract_pool_liquidity(data, &source_dex);
        let target_liquidity = self.extract_pool_liquidity(data, &target_dex);
        
        // Calculate optimal trade size based on liquidity and price difference
        let optimal_trade_size = self.calculate_optimal_size(
            source_liquidity, 
            target_liquidity, 
            price_diff_percentage
        );
        
        // Calculate price impact for this trade size
        let source_price_impact = self.calculate_price_impact(optimal_trade_size, source_liquidity);
        let target_price_impact = self.calculate_price_impact(optimal_trade_size, target_liquidity);
        
        // Total price impact
        let total_price_impact = source_price_impact + target_price_impact;
        
        // Check if price impact is acceptable
        if total_price_impact > self.config.max_price_impact_percent {
            debug!(target: "arbitrage_evaluator", 
                   total_price_impact = total_price_impact,
                   max_impact = self.config.max_price_impact_percent,
                   "Price impact exceeds maximum allowed");
            return Ok(None);
        }
        
        // Estimate gas costs
        let estimated_instruction_count = 8; // Realistic estimate for arbitrage transaction
        let estimated_gas_cost = self.estimate_gas_cost(estimated_instruction_count);
        
        // Calculate expected profit
        let gross_profit = optimal_trade_size * (price_diff_percentage / 100.0);
        let net_profit = gross_profit - estimated_gas_cost;
        
        // Check if net profit is positive
        if net_profit <= 0.0 {
            debug!(target: "arbitrage_evaluator", 
                   gross_profit = gross_profit,
                   gas_cost = estimated_gas_cost,
                   net_profit = net_profit,
                   "Arbitrage not profitable after gas costs");
            return Ok(None);
        }
        
        // Determine risk level
        let risk_level = self.determine_risk_level(
            price_diff_percentage, 
            source_liquidity.min(target_liquidity),
            total_price_impact
        );
        
        // Create metadata
        let metadata = ArbitrageMetadata {
            source_dex,
            target_dex,
            token_path: vec![token_pair.0.clone(), token_pair.1.clone(), token_pair.0.clone()],
            price_difference_percent: price_diff_percentage,
            estimated_gas_cost_usd: estimated_gas_cost,
            optimal_trade_size_usd: optimal_trade_size,
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
        
        // Create the opportunity
        let opportunity = MevOpportunity {
            strategy: MevStrategy::Arbitrage,
            estimated_profit: net_profit,
            confidence: 0.8, // Can be refined based on data quality
            risk_level,
            required_capital: 0.0,
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