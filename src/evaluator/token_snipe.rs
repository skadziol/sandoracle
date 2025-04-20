use crate::evaluator::{MevStrategy, MevOpportunity, MevStrategyEvaluator, RiskLevel};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use tracing::{debug, info, trace, warn};
use std::collections::HashMap;
use rand;
use anyhow::anyhow;
use crate::market_data::{MarketDataCollector, MarketData};
use crate::types::{DecodedInstructionInfo, DecodedTransactionDetails};
use solana_sdk::pubkey::Pubkey;
use std::str::FromStr;

/// Configuration for token sniping opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSnipeConfig {
    /// Minimum initial liquidity (in USD)
    pub min_initial_liquidity_usd: f64,
    /// Maximum amount to invest per token (in USD)
    pub max_investment_per_token_usd: f64,
    /// Minimum expected return multiplier
    pub min_return_multiplier: f64,
    /// Maximum time to hold token (in seconds)
    pub max_hold_time_seconds: u64,
    /// Minimum market cap for token (in USD)
    pub min_market_cap_usd: f64,
    /// Maximum percentage of token supply to acquire
    pub max_token_supply_percent: f64,
    /// Number of blocks to wait before buying after pool creation
    pub blocks_to_wait: u64,
    /// Enable multi-stage exit strategy
    pub use_staged_exit: bool,
}

impl Default for TokenSnipeConfig {
    fn default() -> Self {
        Self {
            min_initial_liquidity_usd: 20000.0,  // $20k minimum liquidity
            max_investment_per_token_usd: 1000.0, // $1k maximum investment
            min_return_multiplier: 2.0,          // 2x minimum expected return
            max_hold_time_seconds: 3600,         // 1 hour maximum hold time
            min_market_cap_usd: 100000.0,        // $100k minimum market cap
            max_token_supply_percent: 0.01,      // 1% maximum token supply
            blocks_to_wait: 1,                   // Wait 1 block before buying
            use_staged_exit: true,               // Enable staged exit
        }
    }
}

/// Take profit stage for exit strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakeProfitStage {
    /// Price multiplier that triggers this stage
    pub trigger_multiplier: f64,
    /// Percentage of position to sell at this stage
    pub percentage_to_sell: f64,
}

/// Exit strategy for token sniping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenExitStrategy {
    /// Multi-stage take profit levels
    pub take_profit_stages: Vec<TakeProfitStage>,
    /// Stop loss percentage (from initial price)
    pub stop_loss_percentage: f64,
    /// Trailing stop percentage (from highest price)
    pub trailing_stop_percentage: f64,
    /// Maximum time to hold token (in seconds)
    pub max_hold_time_seconds: u64,
    /// Initial price at time of entry
    pub initial_price: f64,
    /// Amount invested
    pub investment_amount: f64,
}

/// Metadata for token sniping opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSnipeMetadata {
    /// Token address
    pub token_address: String,
    /// Token name/symbol
    pub token_symbol: String,
    /// DEX where token is listed
    pub dex: String,
    /// Initial liquidity (in USD)
    pub initial_liquidity_usd: f64,
    /// Token price at detection
    pub initial_price_usd: f64,
    /// Initial market cap (in USD)
    pub initial_market_cap_usd: f64,
    /// Proposed investment amount (in USD)
    pub proposed_investment_usd: f64,
    /// Expected return multiplier
    pub expected_return_multiplier: f64,
    /// Maximum position as percentage of supply
    pub max_position_percent: f64,
    /// Optimal hold time estimate (in seconds)
    pub optimal_hold_time_seconds: u64,
    /// Percentage of circulating supply to acquire
    pub acquisition_supply_percent: f64,
    /// Exit strategy
    pub exit_strategy: TokenExitStrategy,
    /// Is pending/mempool transaction
    pub is_pending: bool,
    /// Number of blocks to wait before buying
    pub blocks_to_wait: u64,
    /// Security checks passed
    pub security_checks_passed: bool,
}

/// Evaluator for token sniping opportunities
pub struct TokenSnipeEvaluator {
    config: TokenSnipeConfig,
}

impl TokenSnipeEvaluator {
    /// Create a new token snipe evaluator with default configuration
    pub fn new() -> Self {
        Self {
            config: TokenSnipeConfig::default(),
        }
    }
    
    /// Create a new token snipe evaluator with custom configuration
    pub fn with_config(config: TokenSnipeConfig) -> Self {
        Self { config }
    }
    
    /// Calculate optimal investment amount
    fn calculate_investment_amount(&self, initial_liquidity_usd: f64, initial_market_cap_usd: f64) -> f64 {
        // TODO: Use real initial liquidity / market cap when available
        warn!(target: "token_snipe_evaluator", "Using PLACEHOLDER liquidity/MC for investment calculation.");
        let liquidity = initial_liquidity_usd.max(1000.0); // Placeholder floor
        let market_cap = initial_market_cap_usd.max(10000.0); // Placeholder floor
        
        // Start with a percentage of liquidity (1-3%)
        let max_liquidity_based = liquidity * 0.03;
        
        // Cap by config maximum
        let max_amount = f64::min(max_liquidity_based, self.config.max_investment_per_token_usd);
        
        // Ensure we don't exceed max market cap percentage (relative to estimated MC)
        let market_cap_limit = market_cap * self.config.max_token_supply_percent;
        
        // Return minimum of calculated maximum and market cap limit
        f64::min(max_amount, market_cap_limit)
    }
    
    /// Create exit strategy for token snipe
    fn create_exit_strategy(&self, initial_price: f64, investment_amount: f64) -> TokenExitStrategy {
        // Create a multi-stage exit strategy with take-profit and stop-loss
        
        // Basic strategy:
        // 1. Take 25% profit at 2x
        // 2. Take 25% profit at 3x
        // 3. Take 25% profit at 5x 
        // 4. Take final 25% profit at 10x or hold for max_hold_time
        // 5. Stop loss at 20% down
        
        TokenExitStrategy {
            take_profit_stages: vec![
                TakeProfitStage {
                    trigger_multiplier: 2.0,
                    percentage_to_sell: 25.0,
                },
                TakeProfitStage {
                    trigger_multiplier: 3.0,
                    percentage_to_sell: 25.0,
                },
                TakeProfitStage {
                    trigger_multiplier: 5.0,
                    percentage_to_sell: 25.0,
                },
                TakeProfitStage {
                    trigger_multiplier: 10.0,
                    percentage_to_sell: 25.0,
                },
            ],
            stop_loss_percentage: 20.0,
            trailing_stop_percentage: 30.0,
            max_hold_time_seconds: self.config.max_hold_time_seconds,
            initial_price,
            investment_amount,
        }
    }
    
    /// Estimate risk level based on token metrics
    fn determine_risk_level(&self, initial_liquidity_usd: f64, initial_market_cap_usd: f64) -> RiskLevel {
        // Token sniping is inherently high risk
        // TODO: Use real initial liquidity / market cap
        warn!(target: "token_snipe_evaluator", "Using PLACEHOLDER liquidity/MC for risk level determination.");
        let liquidity = initial_liquidity_usd.max(1000.0);
        let market_cap = initial_market_cap_usd.max(10000.0);

        if liquidity < 50000.0 || market_cap < 200000.0 {
            RiskLevel::High
        } else if liquidity < 100000.0 || market_cap < 500000.0 {
            RiskLevel::Medium // Still quite risky
        } else {
            RiskLevel::Medium // Even with decent stats, sniping is Medium risk at best
        }
    }
    
    /// Estimate potential return multiplier based on market data
    fn estimate_return_multiplier(&self, initial_liquidity_usd: f64, initial_market_cap_usd: f64) -> f64 {
        // TODO: Use real initial liquidity / market cap
        warn!(target: "token_snipe_evaluator", "Using PLACEHOLDER liquidity/MC for return estimation.");
        let liquidity = initial_liquidity_usd.max(1000.0);
        let market_cap = initial_market_cap_usd.max(10000.0);

        // Base multiplier - lower base for more realistic expectation
        let base_multiplier = 3.0; 
        
        // Adjust based on liquidity and market cap (inverse relationship)
        let liquidity_factor = 50000.0 / f64::max(liquidity, 10000.0); // Cap effect
        let market_cap_factor = 200000.0 / f64::max(market_cap, 50000.0); // Cap effect
        
        // Combined multiplier calculation (dampened effect)
        (base_multiplier * (1.0 + liquidity_factor * 0.1 + market_cap_factor * 0.1)).min(10.0) // Cap max multiplier
    }
}

#[async_trait]
impl MevStrategyEvaluator for TokenSnipeEvaluator {
    fn strategy_type(&self) -> MevStrategy {
        MevStrategy::TokenSnipe
    }
    
    async fn evaluate(&self, data: &Value) -> Result<Option<MevOpportunity>> {
        trace!(target: "token_snipe_evaluator", "Evaluating potential token snipe opportunity");

        // --- Parse Context Passed from OpportunityEvaluator --- 
        let market_context = data.get("market_context").ok_or_else(|| {
            warn!(target: "token_snipe_evaluator", "Missing 'market_context' in input data");
            anyhow!("Missing 'market_context' in input data")
        })?;
        let decoded_context = data.get("decoded_details"); // Optional

        // Extract SOL price (though not directly used in core snipe logic currently)
        let _sol_usd_price = market_context
            .get("sol_usd_price")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| {
                 warn!(target: "token_snipe_evaluator", "Missing or invalid SOL price in context, using default.");
                 150.0 
            });
            
        // TODO: Extract quote token price (e.g., USDC price) from market_context if available
        let quote_token_usd_price = 1.0; // Still using placeholder

        // --- Attempt to Parse Pool Init Details from Decoded Context --- 
        let mut parsed_new_token_mint: Option<String> = None;
        let mut parsed_quote_token_mint: Option<String> = None;
        let mut parsed_initial_new_token_amount_raw: Option<u64> = None;
        let mut parsed_initial_quote_amount_raw: Option<u64> = None;
        let mut is_pool_init_event = false;
        
        if let Some(decoded_json) = decoded_context {
             if !decoded_json.is_null() { 
                 match serde_json::from_value::<DecodedTransactionDetails>(decoded_json.clone()) {
                    Ok(details) => {
                        if let Some(primary_ix) = details.primary_instruction {
                            // Check if it looks like a pool init instruction
                            if primary_ix.instruction_name.contains("InitializePool") || primary_ix.instruction_name.contains("Initialize") {
                                is_pool_init_event = true;
                                debug!(target: "token_snipe_evaluator", "Parsed pool init details from decoded context");
                                parsed_new_token_mint = primary_ix.token_a_mint; // Assumption
                                parsed_quote_token_mint = primary_ix.token_b_mint; // Assumption
                                parsed_initial_new_token_amount_raw = primary_ix.initial_token_a_amount;
                                parsed_initial_quote_amount_raw = primary_ix.initial_token_b_amount;
                            } else {
                                 trace!(target: "token_snipe_evaluator", ix_name=%primary_ix.instruction_name, "Decoded instruction is not relevant pool init type for sniping");
                            }
                        }
                    },
                    Err(e) => {
                         warn!(target: "token_snipe_evaluator", error=%e, "Failed to deserialize DecodedTransactionDetails");
                    }
                }
            }
        }
        
        // --- Exit if not a relevant pool initialization event --- 
        if !is_pool_init_event {
             trace!(target: "token_snipe_evaluator", "Data did not contain a valid pool initialization event.");
            return Ok(None); // Not a snipe opportunity if no pool init decoded
        }
        
        // --- Use Parsed Details or Fallback to Placeholders --- 
        let new_token_mint = parsed_new_token_mint.unwrap_or_else(|| {
             warn!(target: "token_snipe_evaluator", "Pool init decoded but missing token A mint? Using placeholder.");
            "NEWT0kEnMintAddressP1aceh01der11111111111".to_string()
        });
        let quote_token_mint = parsed_quote_token_mint.unwrap_or_else(||{
             warn!(target: "token_snipe_evaluator", "Pool init decoded but missing token B mint? Using placeholder.");
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string() // USDC
        });
        let initial_new_token_liquidity_raw = parsed_initial_new_token_amount_raw.unwrap_or_else(|| {
             warn!(target: "token_snipe_evaluator", "Pool init decoded but missing token A amount? Using placeholder.");
            1_000_000_000_000u64 // 1M tokens w/ 6 decimals
        });
         let initial_quote_liquidity_raw = parsed_initial_quote_amount_raw.unwrap_or_else(|| {
             warn!(target: "token_snipe_evaluator", "Pool init decoded but missing quote token amount? Using placeholder.");
             20_000_000_000u64 // 20k USDC w/ 6 decimals
        });
        
        let new_token_symbol = "NEWT".to_string(); // Placeholder
        let new_token_decimals = 6; // Placeholder
        let quote_token_decimals = 6; // Placeholder
        let dex = "Raydium".to_string(); // Placeholder
        let tx_hash = data.get("signature").and_then(|v| v.as_str()).unwrap_or("POOL_CREATE_SIG").to_string();
        warn!(target: "token_snipe_evaluator", "Using PLACEHOLDERS for symbol, decimals, DEX, quote price.");
        // --- End Placeholders/Parsing --- 

        // --- Calculations based on potentially parsed initial liquidity --- 
        let initial_quote_liquidity_ui = initial_quote_liquidity_raw as f64 / 10f64.powi(quote_token_decimals);
        let initial_liquidity_usd = initial_quote_liquidity_ui * quote_token_usd_price * 2.0; 
        let initial_new_token_liquidity_ui = initial_new_token_liquidity_raw as f64 / 10f64.powi(new_token_decimals);
        let initial_price_usd = if initial_new_token_liquidity_ui > 0.0 {
            (initial_quote_liquidity_ui * quote_token_usd_price) / initial_new_token_liquidity_ui
        } else { 0.0 };
        let initial_market_cap_usd = initial_liquidity_usd * 5.0; // Still speculative

        // --- Perform checks --- 

        // Check minimum liquidity
        if initial_liquidity_usd < self.config.min_initial_liquidity_usd {
            trace!(target: "token_snipe_evaluator", liq=initial_liquidity_usd, "Initial liquidity too low");
            return Ok(None);
        }

        // Check minimum market cap
        if initial_market_cap_usd < self.config.min_market_cap_usd {
            trace!(target: "token_snipe_evaluator", mc=initial_market_cap_usd, "Initial market cap too low");
            return Ok(None);
        }

        // Calculate optimal investment and expected return (using potentially parsed initial data)
        let investment_amount = self.calculate_investment_amount(initial_liquidity_usd, initial_market_cap_usd);
        let return_multiplier = self.estimate_return_multiplier(initial_liquidity_usd, initial_market_cap_usd);

        // Check minimum return
        if return_multiplier < self.config.min_return_multiplier {
            trace!(target: "token_snipe_evaluator", ret=return_multiplier, "Estimated return too low");
            return Ok(None);
        }

        // Determine risk level (using potentially parsed initial data)
        let risk_level = self.determine_risk_level(initial_liquidity_usd, initial_market_cap_usd);

        // Create exit strategy (using potentially parsed initial price)
        let exit_strategy = self.create_exit_strategy(initial_price_usd, investment_amount);

        // Calculate expected profit
        let expected_profit = investment_amount * (return_multiplier - 1.0);

        // Security checks would ideally be done upstream by OpportunityEvaluator
        // For now, assume they passed unless specific flags are passed in `data`
        let security_checks_passed = data.get("security_passed").and_then(|v| v.as_bool()).unwrap_or(true);

        // Create metadata using potentially parsed initial data
        let metadata = TokenSnipeMetadata {
            token_address: new_token_mint.clone(),
            token_symbol: new_token_symbol.clone(),
            dex,
            initial_liquidity_usd,
            initial_price_usd,
            initial_market_cap_usd,
            proposed_investment_usd: investment_amount,
            expected_return_multiplier: return_multiplier,
            max_position_percent: investment_amount / initial_liquidity_usd.max(1.0) * 100.0,
            optimal_hold_time_seconds: self.config.max_hold_time_seconds,
            acquisition_supply_percent: investment_amount / initial_market_cap_usd.max(1.0) * 100.0,
            exit_strategy,
            is_pending: false,
            blocks_to_wait: self.config.blocks_to_wait,
            security_checks_passed,
        };

        // Create opportunity using potentially parsed mints
        let opportunity = MevOpportunity {
            strategy: MevStrategy::TokenSnipe,
            estimated_profit: expected_profit,
            confidence: 0.65,
            risk_level,
            required_capital: investment_amount,
            execution_time: 1500,
            metadata: serde_json::to_value(metadata)?,
            score: None,
            decision: None,
            involved_tokens: vec![quote_token_mint, new_token_mint],
            allowed_output_tokens: vec![],
            allowed_programs: vec![],
            max_instructions: 5,
        };

        info!(
            target: "token_snipe_evaluator",
            token = new_token_symbol,
            profit = expected_profit,
            "Found potential token snipe opportunity (using placeholders)"
        );

        Ok(Some(opportunity))
    }
    
    async fn validate(&self, opportunity: &MevOpportunity) -> Result<bool> {
        // ... (validation logic largely unchanged, relies on potentially placeholder data) ...
        // TODO: Implement real validation (check if pool exists, liquidity hasn't been rugged etc.)
        Ok(true) // Placeholder validation for now
    }
}

#[cfg(test)]
mod tests {
    // ... tests will need significant updates ...
} 