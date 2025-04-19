use crate::evaluator::{MevStrategy, MevOpportunity, MevStrategyEvaluator, RiskLevel};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use tracing::{debug, info, trace, warn};
use std::str::FromStr;
use std::collections::HashMap;
use uuid;

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
    
    /// Extract transaction details from Solana transactions to look for sandwich opportunities
    pub fn extract_transaction_details(&self, tx_data: &Value) -> Result<Option<SandwichDetails>> {
        // Check if this is a pending transaction from mempool
        let is_pending = tx_data.get("is_pending")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Get DEX information if available
        let dex = tx_data.get("dex")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        debug!(target: "sandwich_evaluator", is_pending = is_pending, dex = ?dex, "Processing transaction");

        // Skip transactions that aren't from supported DEXes
        if let Some(dex_name) = &dex {
            if !SUPPORTED_DEXES.contains(&dex_name.as_str()) {
                trace!(target: "sandwich_evaluator", dex = dex_name, "Unsupported DEX, skipping");
                return Ok(None);
            }
        } else if !is_pending {
            // For non-pending transactions, we need DEX info
            trace!(target: "sandwich_evaluator", "No DEX identified, skipping non-pending transaction");
            return Ok(None);
        }

        // Extract token pair if available
        let token_pair = if let Some(pair) = tx_data.get("token_pair").and_then(|v| v.as_array()) {
            if pair.len() >= 2 {
                Some((
                    pair[0].as_str().unwrap_or_default().to_string(),
                    pair[1].as_str().unwrap_or_default().to_string()
                ))
            } else {
                None
            }
        } else {
            None
        };

        // Extract transaction value
        let transaction_value = self.extract_transaction_value(tx_data)?;
        
        // Skip low-value transactions to avoid wasting resources
        if transaction_value < self.config.min_transaction_value {
            trace!(
                target: "sandwich_evaluator", 
                value = transaction_value,
                min_value = self.config.min_transaction_value,
                "Transaction value too low, skipping"
            );
            return Ok(None);
        }

        // Check for slippage tolerance in transaction data
        let slippage = self.extract_slippage_tolerance(tx_data)?;
        
        // If slippage is too high, sandwich may not be profitable
        if let Some(slip) = slippage {
            if slip > self.config.max_slippage_bps {
                trace!(
                    target: "sandwich_evaluator", 
                    slippage = slip,
                    max_slippage = self.config.max_slippage_bps,
                    "Slippage tolerance too high, skipping"
                );
                return Ok(None);
            }
        }

        // If we've made it this far, construct sandwich details
        let signature = tx_data.get("signature")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
            
        // Create sandwich details from extracted data
        let details = SandwichDetails {
            transaction_hash: signature,
            dex: dex.unwrap_or_else(|| "unknown".to_string()),
            token_pair: token_pair,
            slippage_tolerance: slippage,
            transaction_value,
            is_pending,
            timestamp: chrono::Utc::now(),
        };

        debug!(
            target: "sandwich_evaluator",
            tx_hash = details.transaction_hash,
            dex = details.dex,
            value = details.transaction_value,
            "Found potential sandwich opportunity"
        );
        
        Ok(Some(details))
    }

    /// Extract transaction value from various sources in the transaction data
    fn extract_transaction_value(&self, tx_data: &Value) -> Result<f64> {
        // Try to extract from different sources in priority order
        
        // 1. Try direct value field if available
        if let Some(value) = tx_data.get("transaction_value")
            .and_then(|v| v.as_f64()) 
        {
            return Ok(value);
        }
        
        // 2. Try to extract from instruction data for known DEXes
        if let Some(transaction) = tx_data.get("transaction") {
            if let Some(instruction_data) = self.extract_instruction_data(transaction) {
                // For Jupiter
                if instruction_data.contains("Jupiter") || 
                   instruction_data.contains("JUP") {
                    // Parse instruction data for amount
                    if let Some(amount) = self.parse_jup_instruction_data(&instruction_data) {
                        return Ok(amount);
                    }
                }
                
                // For Orca
                if instruction_data.contains("Orca") {
                    // Parse instruction data for amount
                    if let Some(amount) = self.parse_orca_instruction_data(&instruction_data) {
                        return Ok(amount);
                    }
                }
                
                // Add more DEX-specific parsers as needed
            }
        }
        
        // 3. Fall back to meta.fee if available
        if let Some(fee) = tx_data
            .get("transaction")
            .and_then(|tx| tx.get("meta"))
            .and_then(|meta| meta.get("fee"))
            .and_then(|fee| fee.as_u64())
        {
            // Use fee as a rough proxy for value (multiply by a constant factor)
            // This is very approximate but better than nothing
            return Ok(fee as f64 * 10.0); // Multiply by 10 as a heuristic
        }
        
        // 4. Default value if we couldn't extract
        debug!(target: "sandwich_evaluator", "Could not extract transaction value, using default");
        Ok(1000.0) // Default $1000 value as a safe assumption
    }

    /// Extract slippage tolerance from transaction data
    fn extract_slippage_tolerance(&self, tx_data: &Value) -> Result<Option<u64>> {
        // Try to extract from instruction data
        if let Some(transaction) = tx_data.get("transaction") {
            if let Some(instruction_data) = self.extract_instruction_data(transaction) {
                // Look for slippage patterns in different DEXes
                
                // Jupiter pattern
                if let Some(slippage) = instruction_data.find("slippage")
                    .and_then(|_| {
                        // Find the slippage value after "slippage"
                        // This is a simple approach, in practice you'd parse the actual data structure
                        let slippage_str = &instruction_data["slippage".len()..];
                        slippage_str.find(|c: char| c.is_digit(10))
                            .and_then(|start_idx| {
                                let num_str = &slippage_str[start_idx..];
                                let end_idx = num_str.find(|c: char| !c.is_digit(10))
                                    .unwrap_or(num_str.len());
                                num_str[..end_idx].parse::<u64>().ok()
                            })
                    })
                {
                    return Ok(Some(slippage));
                }
                
                // Add more DEX-specific pattern matching as needed
            }
        }
        
        // Default to None if we couldn't extract slippage
        Ok(None)
    }
    
    /// Helper to extract instruction data from transaction
    fn extract_instruction_data(&self, transaction: &Value) -> Option<String> {
        // Try to get instruction data from message
        transaction
            .get("message")?
            .get("instructions")?
            .as_array()?
            .iter()
            .filter_map(|ix| ix.get("data").and_then(|d| d.as_str()))
            .collect::<Vec<&str>>()
            .join(" ")
            .into()
    }
    
    // DEX-specific instruction data parsers
    fn parse_jup_instruction_data(&self, data: &str) -> Option<f64> {
        // Implementation depends on Jupiter's data format
        // This is a placeholder - would need to be implemented based on actual data format
        None
    }
    
    fn parse_orca_instruction_data(&self, data: &str) -> Option<f64> {
        // Implementation depends on Orca's data format
        // This is a placeholder - would need to be implemented based on actual data format
        None
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
    
    /// Calculate price impact based on trade size and pool liquidity
    fn calculate_price_impact(&self, trade_size_usd: f64, pool_liquidity_usd: f64) -> f64 {
        // Simple price impact formula based on constant product formula
        // For a more accurate calculation, we'd use the specific AMM formula for each DEX
        let trade_ratio = trade_size_usd / pool_liquidity_usd;
        
        // Using a simplified price impact formula: 
        // impact = trade_ratio / (2 - trade_ratio) for small values of trade_ratio
        trade_ratio / (2.0 - trade_ratio.min(0.5)) * 100.0
    }
    
    /// Estimate gas cost for sandwich transaction (frontrun or backrun)
    fn estimate_gas_cost(&self, instruction_count: u64, use_priority_fee: bool, base_priority_fee: u64) -> f64 {
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
        
        // Convert to USD (assuming $100 SOL price as a placeholder)
        // This should be replaced with actual SOL/USD price from market data
        let sol_price_usd = 100.0;
        
        gas_cost_sol * sol_price_usd
    }
    
    /// Estimate profit from a sandwich opportunity
    fn estimate_profit(&self, position_size: f64, victim_tx_size: f64, pool_liquidity: f64, 
                      victim_slippage: f64, frontrun_gas: f64, backrun_gas: f64) -> f64 {
        // Calculate the price impact from our frontrun tx
        let frontrun_impact = self.calculate_price_impact(position_size, pool_liquidity);
        
        // Calculate how much of the victim's slippage we can capture
        // The victim's tx will now face our frontrun impact + their own impact
        let victim_impact = self.calculate_price_impact(victim_tx_size, pool_liquidity);
        
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
        
        // First, check if this is a pending transaction
        let is_pending = data.get("is_pending").and_then(|v| v.as_bool()).unwrap_or(false);
        if !is_pending {
            trace!(target: "sandwich_evaluator", "Transaction is already confirmed, not suitable for sandwich attack");
            return Ok(None);
        }
        
        // Extract transaction details from result
        let tx_details = match self.extract_transaction_details(data)? {
            Some(details) => details,
            None => {
                trace!(target: "sandwich_evaluator", "Could not extract transaction details");
                return Ok(None);
            }
        };

        let tx_hash = tx_details.transaction_hash;
        let dex = tx_details.dex;
        let tx_value = tx_details.transaction_value;
        let slippage_tolerance = tx_details.slippage_tolerance.unwrap_or(100);
        let token_pair = tx_details.token_pair.unwrap_or(("UNKNOWN".to_string(), "UNKNOWN".to_string()));
        
        // Skip if transaction value is outside our target range
        if tx_value < self.config.min_target_tx_value_usd || tx_value > self.config.max_target_tx_value_usd {
            trace!(
                target: "sandwich_evaluator",
                tx_value = tx_value,
                min = self.config.min_target_tx_value_usd,
                max = self.config.max_target_tx_value_usd,
                "Transaction value outside target range"
            );
            return Ok(None);
        }
        
        // Check if victim slippage tolerance is high enough
        if slippage_tolerance < self.config.min_victim_slippage_pct as u64 {
            trace!(
                target: "sandwich_evaluator",
                slippage = slippage_tolerance,
                min = self.config.min_victim_slippage_pct,
                "Victim slippage tolerance too low"
            );
            return Ok(None);
        }
        
        // Get pool liquidity data (from market data if available)
        let pool_liquidity = data.get("pool_liquidity")
            .and_then(|v| v.as_object())
            .and_then(|obj| obj.get(&dex))
            .and_then(|v| v.as_f64())
            .unwrap_or(100000.0 + tx_value * 10.0); // Default if not available
        
        // Skip if pool liquidity is too low
        if pool_liquidity < self.config.min_pool_liquidity_usd {
            trace!(
                target: "sandwich_evaluator",
                pool_liquidity = pool_liquidity,
                min = self.config.min_pool_liquidity_usd,
                "Pool liquidity too low"
            );
            return Ok(None);
        }
        
        // Calculate optimal position size
        let position_size = self.calculate_optimal_position(tx_value, pool_liquidity);
        
        // Calculate price impact
        let price_impact = self.calculate_price_impact(position_size, pool_liquidity);
        
        // Extract priority fee from victim transaction (for realistic priority fee bumping)
        let base_priority_fee = data.get("transaction")
            .and_then(|tx| tx.get("meta"))
            .and_then(|meta| meta.get("computeUnits"))
            .and_then(|cu| cu.as_object())
            .and_then(|cu_obj| cu_obj.get("priorityFee"))
            .and_then(|fee| fee.as_u64())
            .unwrap_or(500); // Default 500 microlamports/CU
        
        // Estimate gas costs (frontrun needs priority fee, backrun doesn't)
        let frontrun_gas = self.estimate_gas_cost(8, true, base_priority_fee);  // 8 instructions for frontrun
        let backrun_gas = self.estimate_gas_cost(8, false, 0);   // 8 instructions for backrun
        
        // Estimate profit
        let estimated_profit = self.estimate_profit(
            position_size, 
            tx_value, 
            pool_liquidity,
            slippage_tolerance as f64,
            frontrun_gas,
            backrun_gas
        );
        
        // Skip if estimated profit is too low
        if estimated_profit < self.config.min_profit_usd {
            trace!(
                target: "sandwich_evaluator",
                profit = estimated_profit,
                min = self.config.min_profit_usd,
                "Estimated profit too low"
            );
            return Ok(None);
        }
        
        // Determine risk level
        let risk_level = self.determine_risk_level(tx_value, pool_liquidity, price_impact, slippage_tolerance as f64);
        
        // Calculate priority fee (in microlamports per CU)
        let priority_fee = (base_priority_fee as f64 * self.config.priority_fee_multiplier) as u64;
        
        // Create metadata
        let metadata = SandwichMetadata {
            dex: dex.clone(),
            target_tx_hash: tx_hash.clone(),
            target_tx_value_usd: tx_value,
            token_pair: token_pair.clone(),
            pool_liquidity_usd: pool_liquidity,
            price_impact_pct: price_impact,
            optimal_position_size_usd: position_size,
            frontrun_slippage_pct: price_impact * 0.5,  // Estimate: half of the price impact
            backrun_slippage_pct: price_impact * 0.3,   // Estimate: 30% of the price impact
            victim_slippage_tolerance_pct: slippage_tolerance as f64,
            frontrun_gas_cost_usd: frontrun_gas,
            backrun_gas_cost_usd: backrun_gas,
            priority_fee,
            estimated_compute_units: 8 * self.config.avg_instruction_cu, // 8 instructions on average
            front_run_amount: position_size,
            time_window_ms: 1000, // 1 second window
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        // Create opportunity
        let opportunity = MevOpportunity {
            strategy: MevStrategy::Sandwich,
            estimated_profit,
            confidence: 0.8, // High confidence with mempool data
            risk_level,
            required_capital: position_size,
            execution_time: 1000, // 1 second estimated execution time
            involved_tokens: vec![token_pair.0.clone(), token_pair.1.clone()],
            allowed_output_tokens: vec![token_pair.0.clone(), token_pair.1.clone()],
            allowed_programs: vec![], // No program restrictions
            max_instructions: 16, // Allow up to 16 instructions (8 for frontrun, 8 for backrun)
            metadata: serde_json::to_value(metadata)?,
            score: None,
            decision: None,
        };
        
        info!(
            target: "sandwich_evaluator",
            tx_hash = tx_hash,
            dex = dex,
            token_pair = ?token_pair,
            victim_tx = tx_value,
            position = position_size,
            profit = estimated_profit,
            "Found potential sandwich opportunity"
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
    use super::*;

    #[tokio::test]
    async fn test_sandwich_evaluation() {
        let config = SandwichConfig::default();
        let evaluator = SandwichEvaluator::with_config(config);

        let test_data = serde_json::json!({
            "is_pending": true,
            "transaction": {
                "signature": "5KtPn1LGuxhFRGB1RNYJpGt1zLpco4root1UNvKSTuVgCsS4QRFyGbZm5zpiZ2zRrDZzP2",
                "meta": {
                    "preBalances": [100000000000, 0, 0],
                    "postBalances": [99900000000, 0, 0],
                    "logMessages": [
                        "Program JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4 invoke",
                        "Program log: Transfer So11111111111111111111111111111111111111112",
                        "Program log: Transfer EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
                    ],
                    "computeUnits": {
                        "priorityFee": 1000
                    }
                },
                "message": {
                    "accountKeys": [
                        "vines1vzrYbzLMRdu58ou5XTby4qAqVRLmqo36NKPTg",
                        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
                        "So11111111111111111111111111111111111111112",
                        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
                    ],
                    "instructions": [
                        {
                            "programId": "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
                            "accounts": [
                                "vines1vzrYbzLMRdu58ou5XTby4qAqVRLmqo36NKPTg", 
                                "So11111111111111111111111111111111111111112",
                                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
                            ],
                            "data": "base64+R0VUX1NXQVBfUFJJQ0U6OkMxaGpiMkYwVm1GMWJIUUtZMjl1YzNSaGJuUWdiMkpxWldOMFZtRjFiSFFLWTI5dWMzUmhiblFnYjJKcVpXTjBWbUYxYkhRS1lXMXZkVzUwU1c0Z0xtSmhZM1JoYldWU1lYUnBidXR4QzJ4emRGQnlhV05sU1c0Z0xtSmhZM1JoYldWU1lYUnBibmtLY0dsd1JtVmxJRE1LYldsdVFXMXZkVzUwUVdOMFpXRnNJQzV3Y21WMlpXNTBWMmwwYUdSeVlYZGljblJ3SW1FME9USXpObVEwWkMweU56ZGhMVFF3WVdRdFlqUXlNUzA1TWpFNE5qWTVOekl3Tm1ZaUMybXBibFp5YzJsdVp5QXdS"
                        }
                    ]
                }
            },
            "pool_liquidity": {
                "Jupiter": 500000.0
            }
        });

        let result = evaluator.evaluate(&test_data).await.unwrap();
        assert!(result.is_some());

        if let Some(opportunity) = result {
            assert_eq!(opportunity.strategy, MevStrategy::Sandwich);
            assert!(opportunity.estimated_profit > 0.0);
            
            // Extract metadata
            let metadata: SandwichMetadata = serde_json::from_value(opportunity.metadata).unwrap();
            assert_eq!(metadata.dex, "Jupiter");
            assert_eq!(metadata.token_pair, ("SOL".to_string(), "USDC".to_string()));
        }
    }
    
    #[tokio::test]
    async fn test_price_impact_calculation() {
        let evaluator = SandwichEvaluator::new();
        
        // Test with various trade sizes and liquidity values
        assert!(evaluator.calculate_price_impact(1000.0, 100000.0) < 1.0); // Small impact
        assert!(evaluator.calculate_price_impact(10000.0, 100000.0) > 5.0); // Larger impact
    }
    
    #[tokio::test]
    async fn test_profit_estimation() {
        let evaluator = SandwichEvaluator::new();
        
        // Test with various scenarios
        let profit = evaluator.estimate_profit(
            5000.0,    // position size
            10000.0,   // victim tx size
            200000.0,  // pool liquidity
            1.0,       // 1% victim slippage
            10.0,      // frontrun gas
            5.0        // backrun gas
        );
        
        // Should be profitable
        assert!(profit > 0.0);
    }
} 