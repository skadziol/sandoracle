use crate::error::{Result, SandoError, StrategyErrorKind};
use crate::evaluator::{MevOpportunity, MevStrategy, StrategyExecutionService};
use solana_sdk::transaction::Transaction;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::instruction::Instruction;
use solana_sdk::message::Message;
use std::str::FromStr;
use tracing::{info, debug, warn, error};
use serde::{Serialize, Deserialize};
use crate::executor::TransactionExecutor;
use async_trait::async_trait;
use anyhow;
use listen_engine::jup::Jupiter;

/// Represents an arbitrage step between tokens on a specific DEX
#[derive(Debug, Clone)]
pub struct ArbitrageStep {
    /// DEX program ID where the swap will be executed
    pub dex_program_id: Pubkey,
    /// Input token mint address
    pub input_token_mint: Pubkey,
    /// Output token mint address
    pub output_token_mint: Pubkey,
    /// Amount of input token to swap
    pub input_amount: u64,
    /// Minimum amount of output token expected (for slippage control)
    pub min_output_amount: u64,
    /// Pool address (if applicable)
    pub pool_address: Option<Pubkey>,
    /// Additional accounts needed for this DEX swap
    pub additional_accounts: Vec<Pubkey>,
}

/// Complete arbitrage path with multiple steps
#[derive(Debug, Clone)]
pub struct ArbitragePath {
    pub steps: Vec<ArbitrageStep>,
}

/// Represents a sandwich attack execution
#[derive(Debug, Clone)]
pub struct SandwichAttack {
    /// DEX program ID where the sandwich will be executed
    pub dex_program_id: Pubkey,
    /// Pool address where the sandwich will occur
    pub pool_address: Pubkey,
    /// Input token mint (token being spent)
    pub input_token_mint: Pubkey,
    /// Output token mint (token being received)
    pub output_token_mint: Pubkey,
    /// Amount for front-run transaction
    pub front_run_amount: u64,
    /// Amount for back-run transaction
    pub back_run_amount: u64,
    /// Target transaction to sandwich
    pub target_tx_hash: String,
    /// Minimum amount out for front-run (slippage control)
    pub min_front_run_out: u64,
    /// Minimum amount out for back-run (slippage control)
    pub min_back_run_out: u64,
}

/// Represents a token snipe execution
#[derive(Debug, Clone)]
pub struct TokenSnipe {
    /// DEX program ID where the token will be sniped
    pub dex_program_id: Pubkey,
    /// Pool address where the token is listed
    pub pool_address: Pubkey,
    /// Input token mint (token being spent, usually a stablecoin)
    pub input_token_mint: Pubkey,
    /// Output token mint (token being sniped)
    pub output_token_mint: Pubkey,
    /// Amount of input token to spend
    pub input_amount: u64,
    /// Minimum amount of output token expected (for slippage control)
    pub min_output_amount: u64,
}

/// Metadata specific to arbitrage opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageMetadata {
    /// The path of tokens in the arbitrage
    pub token_path: Vec<String>,
    /// Expected prices at each step
    pub prices: Vec<f64>,
    /// Liquidity available at each step
    pub liquidity: Vec<f64>,
    /// Estimated price impact at each step
    pub price_impacts: Vec<f64>,
    /// DEXes involved in the arbitrage
    pub dexes: Vec<String>,
}

/// Metadata specific to sandwich opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandwichMetadata {
    /// Target transaction hash
    pub target_tx_hash: String,
    /// Token pair (input, output)
    pub token_pair: (String, String),
    /// Target transaction size
    pub target_tx_size: f64,
    /// Front run amount
    pub front_run_amount: f64,
    /// Back run amount
    pub back_run_amount: f64,
    /// Front run impact
    pub front_run_impact: f64,
    /// Back run impact
    pub back_run_impact: f64,
    /// DEX name
    pub dex: String,
    /// Gas cost estimate
    pub gas_cost: f64,
}

/// Metadata specific to token snipe opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSnipeMetadata {
    /// Token address
    pub token_address: String,
    /// Token name
    pub token_name: String,
    /// Initial price
    pub initial_price: f64,
    /// Social media mentions
    pub social_mentions: u32,
    /// Launch time
    pub launch_time: i64,
    /// Trading volume
    pub volume: f64,
    /// Liquidity amount
    pub liquidity: f64,
    /// Creator address
    pub creator: String,
    /// Gas cost estimate
    pub gas_cost: f64,
    /// DEX where token is listed
    pub dex: String,
}

/// Implements strategy-specific execution logic
pub struct StrategyExecutor {
    executor: TransactionExecutor,
}

impl StrategyExecutor {
    /// Creates a new StrategyExecutor
    pub fn new(executor: TransactionExecutor) -> Self {
        Self { executor }
    }

    /// Executes an MEV opportunity
    pub async fn execute_opportunity(&self, opportunity: &MevOpportunity) -> Result<solana_sdk::signature::Signature> {
        info!(strategy = ?opportunity.strategy, "Executing MEV opportunity");
        
        // Build transaction based on strategy type
        let transaction = match opportunity.strategy {
            MevStrategy::Arbitrage => self.build_arbitrage_transaction(opportunity).await?,
            MevStrategy::Sandwich => self.build_sandwich_transaction(opportunity).await?,
            MevStrategy::TokenSnipe => self.build_token_snipe_transaction(opportunity).await?,
        };
        
        // Simulate the transaction first
        let simulation_result = self.executor.simulate_transaction(opportunity, &transaction).await?;
        
        // Check if simulation was successful and profitable
        if !simulation_result.is_simulation_successful {
            return Err(SandoError::Strategy {
                kind: StrategyErrorKind::SimulationFailed,
                message: simulation_result.error.unwrap_or_else(|| "Unknown simulation error".to_string()),
            });
        }
        
        if !simulation_result.is_profitable {
            return Err(SandoError::Strategy {
                kind: StrategyErrorKind::Unprofitable,
                message: format!(
                    "Strategy not profitable. Estimated profit: {} SOL (${} USD), Gas cost: {} lamports",
                    simulation_result.estimated_profit_sol,
                    simulation_result.estimated_profit_usd,
                    simulation_result.estimated_gas_cost
                ),
            });
        }
        
        if !simulation_result.safety_checks_passed {
            return Err(SandoError::Strategy {
                kind: StrategyErrorKind::SafetyCheckFailed,
                message: "Safety checks failed during simulation".to_string(),
            });
        }
        
        // Execute the transaction
        info!(
            strategy = ?opportunity.strategy,
            estimated_profit_sol = simulation_result.estimated_profit_sol,
            estimated_profit_usd = simulation_result.estimated_profit_usd,
            "Executing transaction after successful simulation"
        );
        
        self.executor.execute_transaction(transaction).await
    }

    /// Builds a transaction for arbitrage execution
    async fn build_arbitrage_transaction(&self, opportunity: &MevOpportunity) -> Result<Transaction> {
        info!("Building arbitrage transaction");
        
        // Parse opportunity metadata
        let metadata: ArbitrageMetadata = serde_json::from_value(opportunity.metadata.clone())?;
        
        // --- Use Jupiter API for Arbitrage ---
        // For simplicity, we'll execute the first step of the arbitrage path using Jupiter.
        // A full implementation might chain multiple Jupiter swaps or use more complex routing.
        if metadata.token_path.len() < 2 {
            return Err(SandoError::Strategy {
                kind: StrategyErrorKind::InvalidParameters,
                message: "Arbitrage path must contain at least 2 tokens".to_string(),
            });
        }

        let input_token = &metadata.token_path[0];
        let output_token = &metadata.token_path[1];
        // Estimate amount based on opportunity data - Needs refinement based on actual opportunity structure
        // Placeholder: use estimated profit or a fixed amount
        let amount_in = opportunity.estimated_profit.max(1_000_000.0) as u64; // Example: Use 1 token unit or estimated profit

        info!(input = %input_token, output = %output_token, amount = amount_in, "Fetching Jupiter quote for arbitrage step 1");

        let quote_response = Jupiter::fetch_quote(input_token, output_token, amount_in).await
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to fetch Jupiter quote: {}", e),
            })?;
        
        info!(quote = ?quote_response, "Received Jupiter quote");

        let versioned_tx = Jupiter::swap(quote_response, &self.executor.signer_pubkey()).await
             .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to get Jupiter swap transaction: {}", e),
            })?;

        // Convert VersionedTransaction to legacy Transaction if needed, or handle appropriately
        // For now, assume we can extract the message or instructions if TransactionExecutor expects legacy
         match versioned_tx {
            solana_sdk::transaction::VersionedTransaction { message, signatures } => {
                 // Attempt to create a legacy transaction from the versioned message
                 if let solana_sdk::message::VersionedMessage::Legacy(msg) = message {
                     Ok(Transaction::new_unsigned(msg))
                 } else {
                     // Handle non-legacy transactions - maybe TransactionExecutor needs updating?
                     // For now, return an error indicating incompatibility
                     Err(SandoError::Strategy {
                         kind: StrategyErrorKind::UnsupportedTransactionVersion,
                         message: "Jupiter API returned a non-legacy transaction version, which is not currently supported by TransactionExecutor.".to_string(),
                     })
                 }
            }
        }
    }

    /// Builds transactions for sandwich attack execution
    async fn build_sandwich_transaction(&self, opportunity: &MevOpportunity) -> Result<Transaction> {
        info!("Building sandwich transaction (front-run only)");
        
        // Parse opportunity metadata
        let metadata: SandwichMetadata = serde_json::from_value(opportunity.metadata.clone())?;

        // --- Use Jupiter API for Sandwich Front-Run ---
        let (input_token, output_token) = metadata.token_pair;
        // Convert UI amount from metadata to raw amount (assuming 6 decimals for simplicity)
        let amount_in = (metadata.front_run_amount * 1_000_000.0) as u64; // TODO: Use actual decimals

        info!(input = %input_token, output = %output_token, amount = amount_in, "Fetching Jupiter quote for sandwich front-run");

        let quote_response = Jupiter::fetch_quote(&input_token, &output_token, amount_in).await
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to fetch Jupiter quote for front-run: {}", e),
            })?;

        info!(quote = ?quote_response, "Received Jupiter quote for front-run");

        let versioned_tx = Jupiter::swap(quote_response, &self.executor.signer_pubkey()).await
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to get Jupiter swap transaction for front-run: {}", e),
            })?;

        // Convert VersionedTransaction to legacy Transaction
         match versioned_tx {
            solana_sdk::transaction::VersionedTransaction { message, signatures } => {
                 if let solana_sdk::message::VersionedMessage::Legacy(msg) = message {
                     Ok(Transaction::new_unsigned(msg))
                 } else {
                     Err(SandoError::Strategy {
                         kind: StrategyErrorKind::UnsupportedTransactionVersion,
                         message: "Jupiter API returned a non-legacy transaction version for sandwich front-run.".to_string(),
                     })
                 }
            }
        }
    }

    /// Builds a transaction for token snipe execution
    async fn build_token_snipe_transaction(&self, opportunity: &MevOpportunity) -> Result<Transaction> {
        info!("Building token snipe transaction");

        // Parse opportunity metadata
        let metadata: TokenSnipeMetadata = serde_json::from_value(opportunity.metadata.clone())?;

        // --- Use Jupiter API for Token Snipe ---
        // Assume input is usually SOL or USDC, output is the target token
        // TODO: Determine input token dynamically (e.g., from config or wallet balance)
        let input_token = "So11111111111111111111111111111111111111112"; // Example: SOL
        let output_token = &metadata.token_address;
        // Convert UI amount from metadata to raw amount (assuming 9 decimals for SOL)
        let amount_in = (metadata.volume.min(1.0) * 1_000_000_000.0) as u64; // Example: Spend 1 SOL or less

        info!(input = %input_token, output = %output_token, amount = amount_in, "Fetching Jupiter quote for token snipe");

        let quote_response = Jupiter::fetch_quote(input_token, output_token, amount_in).await
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to fetch Jupiter quote for snipe: {}", e),
            })?;

        info!(quote = ?quote_response, "Received Jupiter quote for snipe");

        let versioned_tx = Jupiter::swap(quote_response, &self.executor.signer_pubkey()).await
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to get Jupiter swap transaction for snipe: {}", e),
            })?;

        // Convert VersionedTransaction to legacy Transaction
         match versioned_tx {
            solana_sdk::transaction::VersionedTransaction { message, signatures } => {
                 if let solana_sdk::message::VersionedMessage::Legacy(msg) = message {
                     Ok(Transaction::new_unsigned(msg))
                 } else {
                     Err(SandoError::Strategy {
                         kind: StrategyErrorKind::UnsupportedTransactionVersion,
                         message: "Jupiter API returned a non-legacy transaction version for token snipe.".to_string(),
                     })
                 }
            }
        }
    }

    /// Returns the underlying transaction executor for testing
    #[cfg(test)]
    pub fn get_executor(&self) -> &TransactionExecutor {
        &self.executor
    }
}

#[async_trait]
impl StrategyExecutionService for StrategyExecutor {
    async fn execute_opportunity(&self, opportunity: &MevOpportunity) -> anyhow::Result<solana_sdk::signature::Signature> {
        // Convert SandoError to anyhow::Error
        self.execute_opportunity(opportunity).await.map_err(|e| anyhow::anyhow!("{}", e))
    }
} 