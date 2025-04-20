use crate::evaluator::{MevStrategy, MevOpportunity, MevStrategyEvaluator, RiskLevel};
use anyhow::Result;
use anyhow::anyhow;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use tracing::{debug, info, trace, warn};
use std::str::FromStr;
use std::collections::HashMap;
use uuid;
use super::utils; // Import the new utils module
use crate::types::{DecodedInstructionInfo, DecodedTransactionDetails}; // Path should be correct now
use solana_sdk::pubkey::Pubkey;
use crate::executor::TransactionExecutor; // Assuming this is needed
use crate::market_data::{MarketDataCollector, MarketData}; // Import necessary types

// Define the supported DEXes list
const SUPPORTED_DEXES: &[&str] = &["Jupiter", "Orca", "Raydium", "Unknown DEX", "unknown"];

/// Configuration for sandwich trading opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandwichConfig {
    /// Minimum transaction value to target (in USD)
    pub min_target_tx_value_usd: f64,
    /// Maximum transaction value to target (in USD)
    pub max_target_tx_value_usd: f64,
    /// Minimum pool liquidity required (in USD)
    pub min_pool_liquidity_usd: f64,
    /// Minimum estimated profit (in USD)
    pub min_profit_usd: f64,
    /// Maximum capital to use (in USD)
    pub max_capital_usd: f64,
    /// Maximum position size as percentage of pool liquidity
    pub max_position_pct: f64,
    /// Maximum position multiplier relative to victim tx
    pub max_victim_multiplier: f64,
    /// Minimum victim slippage tolerance to target (%)
    pub min_victim_slippage_pct: f64,
    /// Gas price in lamports (for profit calculation)
    pub gas_price_lamports: u64,
    /// Average instruction cost in compute units
    pub avg_instruction_cu: u64,
    /// Priority fee to ensure frontrun position (in microlamports per CU)
    pub priority_fee_multiplier: f64,
    /// Minimum transaction value to consider (in USD)
    pub min_transaction_value: f64,
    /// Maximum slippage tolerance in basis points
    pub max_slippage_bps: u64,
}

impl Default for SandwichConfig {
    fn default() -> Self {
        Self {
            min_target_tx_value_usd: 1000.0,   // $1,000 minimum to target
            max_target_tx_value_usd: 100000.0, // $100,000 maximum to target
            min_pool_liquidity_usd: 50000.0,   // $50,000 minimum pool liquidity
            min_profit_usd: 50.0,              // $50 minimum profit
            max_capital_usd: 10000.0,          // $10,000 maximum capital
            max_position_pct: 0.05,            // 5% of pool liquidity
            max_victim_multiplier: 3.0,        // Up to 3x the victim's transaction value
            min_victim_slippage_pct: 1.0,      // At least 1% slippage tolerance
            gas_price_lamports: 10000,         // Gas price in lamports
            avg_instruction_cu: 200000,        // Average instruction cost in compute units
            priority_fee_multiplier: 1.5,      // 1.5x the victim's priority fee
            min_transaction_value: 1000.0,      // $1,000 minimum to target
            max_slippage_bps: 1000,            // 10% slippage tolerance
        }
    }
}

/// Metadata for sandwich opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandwichMetadata {
    /// DEX where the sandwich will occur
    pub dex: String,
    /// Target transaction hash
    pub target_tx_hash: String,
    /// Target transaction value (in USD)
    pub target_tx_value_usd: f64,
    /// Token pair involved
    pub token_pair: (String, String),
    /// Pool liquidity (in USD)
    pub pool_liquidity_usd: f64,
    /// Estimated price impact (percentage)
    pub price_impact_pct: f64,
    /// Optimal position size (in USD)
    pub optimal_position_size_usd: f64,
    /// Estimated slippage for frontrun transaction
    pub frontrun_slippage_pct: f64,
    /// Estimated slippage for backrun transaction
    pub backrun_slippage_pct: f64,
    /// Victim slippage tolerance
    pub victim_slippage_tolerance_pct: f64,
    /// Estimated gas costs for frontrun tx
    pub frontrun_gas_cost_usd: f64,
    /// Estimated gas costs for backrun tx
    pub backrun_gas_cost_usd: f64,
    /// Estimated priority fee (in microlamports per CU)
    pub priority_fee: u64,
    /// Estimated compute units needed
    pub estimated_compute_units: u64,
    /// Amounts for frontrun transaction
    pub front_run_amount: f64,
    /// Estimated time window (in milliseconds)
    pub time_window_ms: u64,
    /// Timestamp when the transaction was seen
    pub timestamp: i64,
}

/// Represents transaction details for sandwich opportunity evaluation
#[derive(Debug, Clone)]
pub struct SandwichDetails {
    /// Transaction hash
    pub transaction_hash: String,
    /// DEX where the transaction is occurring
    pub dex: String,
    /// Token pair involved
    pub token_pair: Option<(String, String)>,
    /// Slippage tolerance if available
    pub slippage_tolerance: Option<u64>,
    /// Transaction value in USD
    pub transaction_value: f64,
    /// Whether the transaction is pending
    pub is_pending: bool,
    /// Timestamp when the transaction was seen
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Evaluator for sandwich trading opportunities
pub struct SandwichEvaluator {
    config: SandwichConfig,
}

impl SandwichEvaluator {
    /// Create a new sandwich evaluator with default configuration
    pub fn new() -> Self {
        Self {
            config: SandwichConfig::default(),
        }
    }
    
    /// Create a new sandwich evaluator with custom configuration
    pub fn with_config(config: SandwichConfig) -> Self {
        Self { config }
    }
    
    /// Calculate the optimal position size for sandwich attack
    fn calculate_optimal_position(&self, tx_value: f64, pool_liquidity: f64) -> f64 {
        // Calculate different constraints
        let max_by_liquidity = pool_liquidity * self.config.max_position_pct;
        let max_by_config = self.config.max_capital_usd;
        let max_by_victim = tx_value * self.config.max_victim_multiplier;
        
        // Start with a position size based on victim tx
        let base_size = tx_value * 1.5;
        
        // Return the minimum of all constraints to ensure we don't exceed any limit
        f64::min(f64::min(f64::min(max_by_liquidity, max_by_config), max_by_victim), base_size)
    }
    
    /// Estimate gas cost for sandwich transaction (frontrun or backrun)
    fn estimate_gas_cost(&self, instruction_count: u64, use_priority_fee: bool, base_priority_fee: u64, sol_price_usd: f64) -> f64 {
        // Estimate total compute units based on instruction count
        let compute_units = instruction_count * self.config.avg_instruction_cu;
        
        // Base gas cost in lamports
        let base_gas_cost = compute_units * self.config.gas_price_lamports / 100_000;
        
        // Add priority fee if needed (for frontrun transaction)
        let priority_fee = if use_priority_fee {
            // Priority fee is specified in microlamports per compute unit
            // Calculate the total priority fee for all compute units
            let priority_fee_multiplier = if base_priority_fee > 0 {
                (base_priority_fee as f64 * self.config.priority_fee_multiplier) as u64
            } else {
                1000 // Default 1000 microlamports per CU if no base fee detected
            };
            
            (compute_units * priority_fee_multiplier) / 1_000_000
        } else {
            0
        };
        
        // Total gas cost in lamports
        let total_gas_lamports = base_gas_cost + priority_fee;
        
        // Convert to SOL
        let gas_cost_sol = total_gas_lamports as f64 / 1_000_000_000.0;
        
        // Use the passed SOL price
        gas_cost_sol * sol_price_usd
    }
    
    /// Estimate profit from a sandwich opportunity
    fn estimate_profit(&self, position_size: f64, victim_tx_size: f64, pool_liquidity: f64, 
                      victim_slippage: f64, frontrun_gas: f64, backrun_gas: f64) -> f64 {
        // Calculate the price impact from our frontrun tx using the util function
        let frontrun_impact = utils::calculate_price_impact(position_size, pool_liquidity);
        
        // Calculate how much of the victim's slippage we can capture
        // The victim's tx will now face our frontrun impact + their own impact
        let victim_impact = utils::calculate_price_impact(victim_tx_size, pool_liquidity);
        
        // Calculate the profit percentage we can capture from the sandwich
        // Simplified model: we can capture portion of the victim's slippage tolerance
        let capturable_slippage = f64::min(victim_slippage, frontrun_impact + victim_impact) * 0.7;
        
        // Calculate profit (percentage of our position)
        let profit_percentage = capturable_slippage / 100.0;
        
        // Gross profit in USD
        let gross_profit = position_size * profit_percentage;
        
        // Subtract gas costs
        let net_profit = gross_profit - frontrun_gas - backrun_gas;
        
        net_profit
    }
    
    /// Determine risk level for a sandwich opportunity
    fn determine_risk_level(&self, tx_value: f64, pool_liquidity: f64, price_impact: f64, victim_slippage: f64) -> RiskLevel {
        // Extreme values indicate higher risk
        if tx_value > 50000.0 || pool_liquidity < 100000.0 || price_impact > 1.5 {
            return RiskLevel::High;
        }
        
        // Check if the victim's slippage tolerance is close to our frontrun impact
        // If there's little margin, it's more risky
        if price_impact > victim_slippage * 0.8 {
            return RiskLevel::High;
        }
        
        if tx_value > 10000.0 || pool_liquidity < 250000.0 || price_impact > 0.8 {
            return RiskLevel::Medium;
        }
        
        RiskLevel::Low
    }
}

#[async_trait]
impl MevStrategyEvaluator for SandwichEvaluator {
    fn strategy_type(&self) -> MevStrategy {
        MevStrategy::Sandwich
    }
    
    async fn evaluate(&self, data: &Value) -> Result<Option<MevOpportunity>> {
        trace!(target: "sandwich_evaluator", "Evaluating potential sandwich opportunity");
        
        // --- Parse Context Passed from OpportunityEvaluator --- 
        let market_context = data.get("market_context").ok_or_else(|| {
            warn!(target: "sandwich_evaluator", "Missing 'market_context' in input data");
            anyhow!("Missing 'market_context' in input data")
        })?;
        let decoded_context = data.get("decoded_details"); // Optional

        // Extract SOL price (use default if fetch failed)
        let sol_usd_price = market_context
            .get("sol_usd_price")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| {
                 warn!(target: "sandwich_evaluator", "Missing or invalid SOL price in context, using default.");
                 150.0 // Default SOL price
            });

        // Extract pair market data
        let pair_market_data: Option<MarketData> = market_context
            .get("pair_market_data")
            .and_then(|v| serde_json::from_value(v.clone()).ok()); // Deserialize Option<MarketData>

        // Extract liquidity (use 0.0 if MarketData is missing or liquidity is placeholder -1.0)
        let pool_liquidity_usd = pair_market_data
            .as_ref()
            .map_or(0.0, |md| if md.liquidity == -1.0 { 0.0 } else { md.liquidity });

        // --- Parse/Placeholder Victim TX Details --- 
        let is_pending = data.get("is_pending").and_then(|v| v.as_bool()).unwrap_or(true); 
        let tx_hash = data.get("signature").and_then(|v| v.as_str()).unwrap_or("PLACEHOLDER_SIG").to_string();
        let dex = data.get("dex").and_then(|v| v.as_str()).unwrap_or("Jupiter").to_string();
        let victim_priority_fee = data.get("priority_fee").and_then(|v| v.as_u64()).unwrap_or(1000);

        // Defaults / Placeholders
        let mut victim_input_token = "So11111111111111111111111111111111111111112".to_string();
        let mut victim_output_token = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string();
        let mut victim_input_amount_raw: Option<u64> = None; 
        let mut victim_min_output_amount_raw: Option<u64> = None; 

        // Try to overwrite placeholders with decoded details
        if let Some(decoded_json) = decoded_context {
             if !decoded_json.is_null() {
                 match serde_json::from_value::<DecodedTransactionDetails>(decoded_json.clone()) {
                    Ok(details) => {
                        if let Some(primary_ix) = details.primary_instruction {
                            // Use decoded mints if available
                            if let (Some(input_m), Some(output_m)) = (primary_ix.input_mint, primary_ix.output_mint) {
                                victim_input_token = input_m;
                                victim_output_token = output_m;
                                debug!(target: "sandwich_evaluator", tx_sig=%tx_hash, "Using decoded token pair");
                            } else {
                                warn!(target: "sandwich_evaluator", tx_sig=%tx_hash, "Decoded instruction missing mints, using placeholders.");
                            }
                            // Use decoded amounts if available
                            if primary_ix.input_amount.is_some() {
                                victim_input_amount_raw = primary_ix.input_amount;
                                debug!(target: "sandwich_evaluator", tx_sig=%tx_hash, "Using decoded input amount");
                            }
                            if primary_ix.minimum_output_amount.is_some() {
                                victim_min_output_amount_raw = primary_ix.minimum_output_amount;
                                debug!(target: "sandwich_evaluator", tx_sig=%tx_hash, "Using decoded minimum output amount");
                            }
                        }
                    },
                    Err(e) => {
                         warn!(target: "sandwich_evaluator", tx_sig=%tx_hash, error=%e, "Failed to deserialize DecodedTransactionDetails");
                    }
                }
            }
        }
        
        // --- Convert/Calculate Victim TX Values --- 
        // TODO: Need token decimals (fetch via MarketDataCollector?) to convert raw amounts accurately
        let input_token_decimals = 9; // Placeholder (SOL)
        let output_token_decimals = 6; // Placeholder (USDC)
        
        let victim_input_amount_ui = victim_input_amount_raw
            .map(|raw| raw as f64 / 10f64.powi(input_token_decimals))
            .unwrap_or(100.0); // Fallback to placeholder if raw amount not decoded
            
        let input_token_usd_price = if victim_input_token == "So11111111111111111111111111111111111111112" {
            sol_usd_price
        } else {
            // TODO: Fetch USD price for the actual input token if it's not SOL
            warn!(target: "sandwich_evaluator", tx_sig=%tx_hash, token=%victim_input_token, "Cannot get non-SOL input token price yet, using SOL price as estimate.");
            sol_usd_price // Placeholder estimate
        };
        let victim_tx_value_usd = victim_input_amount_ui * input_token_usd_price;

        // Calculate approximate slippage % based on minimum output amount (if available)
        let victim_slippage_pct = if let (Some(min_out_raw), Some(input_raw)) = (victim_min_output_amount_raw, victim_input_amount_raw) {
            // Need approximate output price to compare min_out to expected_out
            // expected_out_raw = input_raw * price (output/input)
            // price = output_usd / input_usd 
            let output_usd_price = input_token_usd_price / pair_market_data.as_ref().map_or(1.0, |md| md.price); // Estimate output price
            let expected_out_ui = victim_input_amount_ui * pair_market_data.as_ref().map_or(1.0, |md| md.price);
            let expected_out_raw = expected_out_ui * 10f64.powi(output_token_decimals);
            
            if expected_out_raw > 0.0 {
                let diff = expected_out_raw - min_out_raw as f64;
                (diff / expected_out_raw * 100.0).max(0.0)
            } else { 0.0 }
        } else {
            warn!(target: "sandwich_evaluator", tx_sig=%tx_hash, "Using PLACEHOLDER slippage (min output not decoded).");
            1.0 // Fallback placeholder 1%
        };
        // --- End Victim TX Calculations --- 

        // --- Start Evaluation Logic --- 
        if !is_pending {
            trace!(target: "sandwich_evaluator", tx_hash=%tx_hash, "Transaction not pending, skipping");
            return Ok(None);
        }

        // Skip if transaction value is outside our target range
        if victim_tx_value_usd < self.config.min_target_tx_value_usd || victim_tx_value_usd > self.config.max_target_tx_value_usd {
            trace!(target: "sandwich_evaluator", tx_hash=%tx_hash, value=victim_tx_value_usd, "Victim TX value outside target range");
            return Ok(None);
        }

        // Skip if victim slippage tolerance is too low (convert bps to pct)
        if victim_slippage_pct < self.config.min_victim_slippage_pct {
             trace!(target: "sandwich_evaluator", tx_hash=%tx_hash, slippage=victim_slippage_pct, "Victim slippage tolerance too low");
            return Ok(None);
        }

        // Use parsed pool liquidity 
        if pool_liquidity_usd < self.config.min_pool_liquidity_usd {
            trace!(target: "sandwich_evaluator", tx_hash=%tx_hash, liquidity=pool_liquidity_usd, "Pool liquidity too low");
            return Ok(None);
        }

        // Calculate optimal position size using potentially real liquidity
        let position_size = self.calculate_optimal_position(victim_tx_value_usd, pool_liquidity_usd);

        // Calculate price impact using potentially real liquidity
        let price_impact = utils::calculate_price_impact(position_size, pool_liquidity_usd);

        // Estimate gas costs using potentially real SOL price
        let frontrun_gas = self.estimate_gas_cost(8, true, victim_priority_fee, sol_usd_price);
        let backrun_gas = self.estimate_gas_cost(8, false, 0, sol_usd_price); 

        // Estimate profit using potentially real liquidity and calculated slippage
        let estimated_profit = self.estimate_profit(
            position_size, 
            victim_tx_value_usd, 
            pool_liquidity_usd,
            victim_slippage_pct, // Use calculated/placeholder slippage %
            frontrun_gas,
            backrun_gas
        );

        // Skip if estimated profit is too low
        if estimated_profit < self.config.min_profit_usd {
            trace!(target: "sandwich_evaluator", tx_hash=%tx_hash, profit=estimated_profit, "Estimated profit too low");
            return Ok(None);
        }

        // Determine risk level using potentially real liquidity and calculated slippage
        let risk_level = self.determine_risk_level(victim_tx_value_usd, pool_liquidity_usd, price_impact, victim_slippage_pct);

        // Calculate final priority fee for frontrun
        let priority_fee_microlamports = (victim_priority_fee as f64 * self.config.priority_fee_multiplier) as u64;

        // Create metadata using potentially parsed/calculated victim details
        let metadata = SandwichMetadata {
            dex: dex.clone(),
            target_tx_hash: tx_hash.clone(),
            target_tx_value_usd: victim_tx_value_usd, // Based on potentially parsed amount
            token_pair: (victim_input_token.clone(), victim_output_token.clone()), // Potentially parsed
            pool_liquidity_usd,
            price_impact_pct: price_impact,
            optimal_position_size_usd: position_size,
            frontrun_slippage_pct: price_impact * 0.5,
            backrun_slippage_pct: price_impact * 0.3,
            victim_slippage_tolerance_pct: victim_slippage_pct, // Use calculated/placeholder %
            frontrun_gas_cost_usd: frontrun_gas,
            backrun_gas_cost_usd: backrun_gas,
            priority_fee: priority_fee_microlamports,
            estimated_compute_units: 8 * self.config.avg_instruction_cu,
            front_run_amount: position_size,
            time_window_ms: 1000,
            timestamp: chrono::Utc::now().timestamp(),
        };

        // Create opportunity using potentially parsed victim details
        let opportunity = MevOpportunity {
            strategy: MevStrategy::Sandwich,
            estimated_profit,
            confidence: 0.7,
            risk_level,
            required_capital: position_size,
            execution_time: 1000,
            involved_tokens: vec![victim_input_token, victim_output_token], // Use potentially parsed pair
            allowed_output_tokens: vec![],
            allowed_programs: vec![],
            max_instructions: 16,
            metadata: serde_json::to_value(metadata)?,
            score: None,
            decision: None,
        };

        info!(
            target: "sandwich_evaluator",
            tx_hash = tx_hash,
            dex = dex,
            victim_tx = victim_tx_value_usd,
            profit = estimated_profit,
            "Found potential sandwich opportunity (using context + placeholders)"
        );

        Ok(Some(opportunity))
    }
    
    async fn validate(&self, opportunity: &MevOpportunity) -> Result<bool> {
        // Extract metadata
        let metadata: SandwichMetadata = match serde_json::from_value(opportunity.metadata.clone()) {
            Ok(meta) => meta,
            Err(e) => {
                warn!(target: "sandwich_evaluator", error = %e, "Failed to parse sandwich metadata during validation");
                return Ok(false);
            }
        };
        
        // For sandwich attacks, validation is critical:
        // 1. The victim transaction must still be in the mempool
        // 2. The transaction must not be stale (too old)
        // 3. The same liquidity pool conditions should still apply
        
        // Check if the opportunity is too old
        let age_ms = chrono::Utc::now()
            .signed_duration_since(chrono::DateTime::<chrono::Utc>::from_timestamp(metadata.timestamp, 0)
                .unwrap_or_else(|| chrono::Utc::now()))
            .num_milliseconds();
            
        if age_ms > metadata.time_window_ms as i64 {
            debug!(
                target: "sandwich_evaluator",
                age_ms = age_ms, 
                max_age = metadata.time_window_ms,
                "Opportunity too old for sandwich attack"
            );
            return Ok(false);
        }
        
        // In a real implementation, you would:
        // 1. Check if the target transaction is still in the mempool via RPC
        // 2. Update pool liquidity data and recalculate profitability
        // 3. Ensure the market conditions haven't changed
        
        debug!(
            target: "sandwich_evaluator",
            tx_hash = &metadata.target_tx_hash,
            token_pair = ?metadata.token_pair,
            "Sandwich opportunity still valid"
        );
        
        // For now, we'll assume it's valid if it's recent enough
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    // ... tests will need significant updates ...
} 