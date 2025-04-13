use crate::error::{Result, SandoError, StrategyErrorKind};
use crate::evaluator::{MevOpportunity, MevStrategy, StrategyExecutionService};
use crate::jupiter_client::Jupiter;
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

/// Enum representing transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransactionStatus {
    Pending,   // Transaction is in mempool
    Confirmed, // Transaction is confirmed
    Failed,    // Transaction failed
    Expired,   // Transaction expired from mempool
    Unknown,   // Status can't be determined
}

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
    /// Create a new strategy executor with the given transaction executor
    pub fn new(executor: TransactionExecutor) -> Self {
        Self { executor }
    }
    
    /// Get a reference to the underlying TransactionExecutor
    pub fn get_executor(&self) -> &TransactionExecutor {
        &self.executor
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
        info!("Building arbitrage transaction for MEV opportunity");
        
        let metadata: ArbitrageMetadata = serde_json::from_value(opportunity.metadata.clone())
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::InvalidParameters,
                message: format!("Failed to parse arbitrage metadata: {}", e)
            })?;
        
        if metadata.token_path.len() < 2 {
            return Err(SandoError::Strategy {
                kind: StrategyErrorKind::InvalidParameters,
                message: "Arbitrage path must contain at least 2 tokens".to_string(),
            });
        }

        // Create a complete cycle by adding the first token at the end if not already there
        let token_path = if metadata.token_path.first() != metadata.token_path.last() {
            let mut complete_path = metadata.token_path.clone();
            complete_path.push(metadata.token_path[0].clone());
            complete_path
        } else {
            metadata.token_path.clone()
        };

        // Get starting amount based on the first token in the path
        // This would typically be a token we already hold
        let input_token = &token_path[0];
        let initial_amount = self.get_optimal_input_amount(input_token, &metadata).await?;
        
        info!(
            input_token = %input_token, 
            initial_amount = initial_amount,
            path_length = token_path.len() - 1,
            "Starting arbitrage execution with multi-step path"
        );

        // Build instruction sets for each step in the arbitrage
        let mut all_instructions: Vec<Instruction> = Vec::new();
        let mut current_amount = initial_amount;
        
        // Execute each step in the arbitrage path
        for i in 0..token_path.len() - 1 {
            let current_token = &token_path[i];
            let next_token = &token_path[i + 1];
            
            // Get the quote for this leg of the arbitrage
            info!(
                step = i + 1,
                from_token = %current_token,
                to_token = %next_token,
                amount = current_amount,
                "Fetching Jupiter quote for arbitrage step"
            );
            
            let quote_response = Jupiter::fetch_quote(current_token, next_token, current_amount).await
                .map_err(|e| SandoError::Strategy {
                    kind: StrategyErrorKind::ApiError,
                    message: format!("Failed to fetch Jupiter quote for step {}: {}", i + 1, e),
                })?;
            
            debug!(
                step = i + 1,
                quote = ?quote_response,
                "Received Jupiter quote for arbitrage step"
            );
            
            // Get minimum output amount with slippage for this step (default 0.5% slippage)
            let slippage = 0.005; // 0.5% slippage
            let out_amount = quote_response.out_amount.parse::<u64>()
                .map_err(|e| SandoError::Strategy {
                    kind: StrategyErrorKind::InvalidParameters,
                    message: format!("Failed to parse output amount: {}", e),
                })?;
            let min_out_amount = ((out_amount as f64) * (1.0 - slippage)) as u64;
            
            // For the next iteration, use out_amount as the input
            current_amount = out_amount;
            
            // Get swap instructions for this step
            let instructions = Jupiter::get_swap_instructions(
                quote_response,
                &self.executor.signer_pubkey(),
                min_out_amount
            ).await
                .map_err(|e| SandoError::Strategy {
                    kind: StrategyErrorKind::ApiError,
                    message: format!("Failed to get swap instructions for step {}: {}", i + 1, e),
                })?;
            
            // Add all instructions from this step
            all_instructions.extend(instructions);
        }
        
        // Calculate expected minimum profit
        let final_amount = current_amount;
        let profit_percentage = (final_amount as f64 / initial_amount as f64) - 1.0;
        
        info!(
            initial_amount = initial_amount,
            final_amount = final_amount,
            profit_percentage = profit_percentage * 100.0,
            "Completed arbitrage path construction"
        );
        
        if final_amount <= initial_amount {
            return Err(SandoError::Strategy {
                kind: StrategyErrorKind::Unprofitable,
                message: format!(
                    "Arbitrage would not be profitable: {} -> {} ({:.2}%)",
                    initial_amount, final_amount, profit_percentage * 100.0
                ),
            });
        }
        
        // Create a single transaction with all instructions
        let message = Message::new(
            &all_instructions,
            Some(&self.executor.signer_pubkey()),
        );
        
        Ok(Transaction::new_unsigned(message))
    }
    
    /// Calculate the optimal input amount for an arbitrage based on metadata
    async fn get_optimal_input_amount(&self, token_mint: &str, metadata: &ArbitrageMetadata) -> Result<u64> {
        // Get token decimals
        let decimals = match token_mint {
            "So11111111111111111111111111111111111111112" => 9, // SOL has 9 decimals
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" => 6, // USDC has 6 decimals
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB" => 6, // USDT has 6 decimals
            _ => {
                // Try to look up decimals on-chain
                if let Ok(pubkey) = Pubkey::from_str(token_mint) {
                    match self.executor.get_mint_decimals(&pubkey).await {
                        Ok(dec) => dec,
                        Err(_) => 6, // Default to 6 if we can't determine
                    }
                } else {
                    6 // Default to 6 if invalid pubkey
                }
            }
        };
        
        // Calculate amount based on price, liquidity and risk assessment
        // For arbitrage, we want to calculate an optimal size that:
        // 1. Is small enough to not significantly impact price (avoiding front-running)
        // 2. Is large enough to make the transaction profitable after fees
        // 3. Takes into account liquidity constraints
        
        // Find the bottleneck liquidity (minimum across all steps)
        let min_liquidity = metadata.liquidity.iter()
            .fold(f64::MAX, |a, b| a.min(*b));
            
        // Start with a percentage of the available liquidity (to avoid high price impact)
        // Use a conservative value (0.5-5% of min liquidity)
        let liquidity_factor = 0.02; // 2% of the bottleneck liquidity
        let ideal_amount = min_liquidity * liquidity_factor;
        
        // Convert to token units
        let amount = (ideal_amount * 10_f64.powi(decimals as i32)) as u64;
        
        // Establish min/max bounds to ensure profitability and manage risk
        let min_amount = 10_u64.pow(decimals as u32) / 100; // Small baseline amount
        let max_amount = match token_mint {
            // Set higher limits for stablecoins
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" => 5_000 * 10_u64.pow(6), // 5,000 USDC max
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB" => 5_000 * 10_u64.pow(6), // 5,000 USDT max
            // More conservative with SOL
            "So11111111111111111111111111111111111111112" => 5 * 10_u64.pow(9), // 5 SOL max
            // Default for other tokens
            _ => 1_000 * 10_u64.pow(decimals as u32), // Reasonable limit based on decimals
        };
        
        let final_amount = amount.clamp(min_amount, max_amount);
        
        info!(
            token = %token_mint,
            decimals = decimals,
            liquidity = min_liquidity,
            calculated_amount = amount,
            final_amount = final_amount,
            "Determined optimal input amount for arbitrage"
        );
        
        Ok(final_amount)
    }

    /// Builds transactions for sandwich attack execution
    async fn build_sandwich_transaction(&self, opportunity: &MevOpportunity) -> Result<Transaction> {
        info!("Building sandwich transaction for MEV opportunity");
        
        let metadata: SandwichMetadata = serde_json::from_value(opportunity.metadata.clone())
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::InvalidParameters,
                message: format!("Failed to parse sandwich metadata: {}", e)
            })?;

        // Extract token pair and amounts from metadata
        let (input_token, output_token) = metadata.token_pair;
        
        // Get input amount using correct token decimals
        let (input_decimals, output_decimals) = self.get_token_decimals_pair(&input_token, &output_token).await?;
        let front_run_amount = self.convert_to_raw_amount(metadata.front_run_amount, input_decimals);
        
        // We need to check if we should be executing the front-run or back-run part of the sandwich
        let target_tx_hash = &metadata.target_tx_hash;
        
        // First, check if the target transaction is still in the mempool (pending)
        let target_status = self.check_transaction_status(target_tx_hash).await?;
        
        match target_status {
            TransactionStatus::Pending => {
                // Execute front-run transaction
                info!(
                    target_tx = %target_tx_hash,
                    "Target transaction is pending, executing front-run part of sandwich"
                );
                
                self.build_sandwich_front_run(&input_token, &output_token, front_run_amount, input_decimals, output_decimals).await
            },
            TransactionStatus::Confirmed => {
                // Execute back-run transaction
                info!(
                    target_tx = %target_tx_hash, 
                    "Target transaction is confirmed, executing back-run part of sandwich"
                );
                
                // For back-run, we swap in the opposite direction using the output from front-run
                // We need to calculate how much of the output token we received from the front-run
                // For simplicity, we'll estimate this based on metadata
                let estimated_output = self.convert_to_raw_amount(
                    metadata.front_run_amount * (1.0 - metadata.front_run_impact), 
                    output_decimals
                );
                
                self.build_sandwich_back_run(&output_token, &input_token, estimated_output, output_decimals, input_decimals).await
            },
            TransactionStatus::Failed | TransactionStatus::Expired => {
                // If target transaction failed or expired, we should abort the sandwich
                // and potentially swap back any tokens we've already purchased
                warn!(
                    target_tx = %target_tx_hash,
                    status = ?target_status,
                    "Target transaction failed or expired, aborting sandwich"
                );
                
                return Err(SandoError::Strategy {
                    kind: StrategyErrorKind::InvalidParameters,
                    message: format!("Target transaction is in invalid state: {:?}", target_status),
                });
            },
            TransactionStatus::Unknown => {
                warn!(
                    target_tx = %target_tx_hash,
                    "Could not determine target transaction status, defaulting to front-run"
                );
                
                self.build_sandwich_front_run(&input_token, &output_token, front_run_amount, input_decimals, output_decimals).await
            },
        }
    }
    
    /// Build the front-run part of a sandwich attack
    async fn build_sandwich_front_run(
        &self, 
        input_token: &str, 
        output_token: &str, 
        amount: u64,
        _input_decimals: u8,
        _output_decimals: u8
    ) -> Result<Transaction> {
        info!(
            input = %input_token,
            output = %output_token,
            amount = amount,
            "Fetching Jupiter quote for sandwich front-run"
        );

        let quote_response = Jupiter::fetch_quote(input_token, output_token, amount).await
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to fetch Jupiter quote for front-run: {}", e),
            })?;

        // Calculate minimum expected output with higher slippage (we expect more price impact)
        // For front-run, we use higher slippage to ensure transaction success
        let slippage = 0.01; // 1% slippage for front-run
        let out_amount = quote_response.out_amount.parse::<u64>()
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::InvalidParameters,
                message: format!("Failed to parse output amount: {}", e),
            })?;
        let min_out_amount = ((out_amount as f64) * (1.0 - slippage)) as u64;
        
        info!(
            expected_output = quote_response.out_amount.parse::<u64>().unwrap_or(0),
            min_output = min_out_amount,
            slippage_percent = slippage * 100.0,
            "Calculated minimum output amount for front-run"
        );
        
        // Get swap instructions
        let instructions = Jupiter::get_swap_instructions(
            quote_response,
            &self.executor.signer_pubkey(),
            min_out_amount
        ).await
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to get Jupiter swap instructions for front-run: {}", e),
            })?;
            
        // Add priority fee to ensure this transaction gets processed quickly
        // before the target transaction
        let compute_budget_instructions = solana_sdk::compute_budget::ComputeBudgetInstruction::set_compute_unit_price(
            5000 // Micro-lamports per compute unit (higher priority fee)
        );
        
        // Combine all instructions
        let mut all_instructions = vec![compute_budget_instructions];
        all_instructions.extend(instructions);
        
        // Create the transaction
        let message = Message::new(
            &all_instructions,
            Some(&self.executor.signer_pubkey()),
        );
        
        Ok(Transaction::new_unsigned(message))
    }
    
    /// Build the back-run part of a sandwich attack
    async fn build_sandwich_back_run(
        &self, 
        input_token: &str, 
        output_token: &str, 
        amount: u64,
        _input_decimals: u8,
        _output_decimals: u8
    ) -> Result<Transaction> {
        info!(
            input = %input_token,
            output = %output_token,
            amount = amount,
            "Fetching Jupiter quote for sandwich back-run"
        );

        let quote_response = Jupiter::fetch_quote(input_token, output_token, amount).await
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to fetch Jupiter quote for back-run: {}", e),
            })?;

        // For back-run, we can use lower slippage since we're selling into a recovering market
        let slippage = 0.003; // 0.3% slippage for back-run
        let out_amount = quote_response.out_amount.parse::<u64>()
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::InvalidParameters,
                message: format!("Failed to parse output amount: {}", e),
            })?;
        let min_out_amount = ((out_amount as f64) * (1.0 - slippage)) as u64;
        
        info!(
            expected_output = quote_response.out_amount.parse::<u64>().unwrap_or(0),
            min_output = min_out_amount,
            slippage_percent = slippage * 100.0,
            "Calculated minimum output amount for back-run"
        );
        
        // Get swap instructions
        let instructions = Jupiter::get_swap_instructions(
            quote_response,
            &self.executor.signer_pubkey(),
            min_out_amount
        ).await
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to get Jupiter swap instructions for back-run: {}", e),
            })?;
            
        // Add priority fee to ensure this transaction gets processed quickly
        // after the target transaction
        let compute_budget_instructions = solana_sdk::compute_budget::ComputeBudgetInstruction::set_compute_unit_price(
            2500 // Micro-lamports per compute unit (medium priority fee)
        );
        
        // Combine all instructions
        let mut all_instructions = vec![compute_budget_instructions];
        all_instructions.extend(instructions);
        
        // Create the transaction
        let message = Message::new(
            &all_instructions,
            Some(&self.executor.signer_pubkey()),
        );
        
        Ok(Transaction::new_unsigned(message))
    }
    
    /// Check the status of a transaction
    async fn check_transaction_status(&self, tx_hash: &str) -> Result<TransactionStatus> {
        // Try to get transaction status from RPC
        let signature = match solana_sdk::signature::Signature::from_str(tx_hash) {
            Ok(sig) => sig,
            Err(_) => {
                return Ok(TransactionStatus::Unknown);
            }
        };
        
        match self.executor.rpc_client.get_signature_status(&signature) {
            Ok(Some(result)) => {
                match result {
                    Ok(_) => Ok(TransactionStatus::Confirmed),
                    Err(_) => Ok(TransactionStatus::Failed),
                }
            },
            Ok(None) => {
                // Transaction not found, check if it's in mempool
                if self.is_in_mempool(tx_hash).await {
                    Ok(TransactionStatus::Pending)
                } else {
                    Ok(TransactionStatus::Expired)
                }
            },
            Err(_) => Ok(TransactionStatus::Unknown),
        }
    }
    
    /// Check if a transaction is in the mempool
    async fn is_in_mempool(&self, _tx_hash: &str) -> bool {
        // This is a placeholder - in a real implementation, we would query a mempool service
        // For now, we'll assume that if we don't have confirmation, it's in the mempool
        // This would need to be replaced with an actual check using listen-engine's
        // mempool data or another source
        true
    }
    
    /// Get the decimal places for a token pair
    async fn get_token_decimals_pair(&self, input_token: &str, output_token: &str) -> Result<(u8, u8)> {
        // Helper function to get decimals for a single token
        async fn get_single_token_decimals(token_mint: &str, executor: &TransactionExecutor) -> Result<u8> {
            match token_mint {
                "So11111111111111111111111111111111111111112" => Ok(9), // SOL has 9 decimals
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" => Ok(6), // USDC has 6 decimals
                "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB" => Ok(6), // USDT has 6 decimals
                _ => {
                    // Try to look up decimals on-chain
                    if let Ok(pubkey) = Pubkey::from_str(token_mint) {
                        executor.get_mint_decimals(&pubkey).await
                    } else {
                        Ok(6) // Default to 6 if invalid pubkey
                    }
                }
            }
        }
        
        let input_decimals = get_single_token_decimals(input_token, &self.executor).await?;
        let output_decimals = get_single_token_decimals(output_token, &self.executor).await?;
        
        Ok((input_decimals, output_decimals))
    }
    
    /// Convert a UI amount to raw token amount
    fn convert_to_raw_amount(&self, ui_amount: f64, decimals: u8) -> u64 {
        (ui_amount * 10_f64.powi(decimals as i32)) as u64
    }
    
    /// Convert a raw token amount to UI amount
    fn convert_to_ui_amount(&self, raw_amount: u64, decimals: u8) -> f64 {
        (raw_amount as f64) / 10_f64.powi(decimals as i32)
    }

    /// Builds a transaction for token snipe execution
    async fn build_token_snipe_transaction(&self, opportunity: &MevOpportunity) -> Result<Transaction> {
        info!("Building token snipe transaction for MEV opportunity");

        let metadata: TokenSnipeMetadata = serde_json::from_value(opportunity.metadata.clone())
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::InvalidParameters,
                message: format!("Failed to parse token snipe metadata: {}", e)
            })?;

        // Determine input token (usually a stable token or SOL)
        // Check wallet balances to determine the best token to use
        let input_token = self.determine_best_input_token().await?;
        let output_token = &metadata.token_address;
        
        info!(
            token_name = %metadata.token_name,
            token_address = %metadata.token_address,
            dex = %metadata.dex,
            liquidity = metadata.liquidity,
            "Preparing to snipe token"
        );
        
        // Get input/output token decimals
        let (input_decimals, output_decimals) = self.get_token_decimals_pair(&input_token, output_token).await?;
        
        // Calculate optimal amount to invest
        let amount_in = self.calculate_snipe_amount(&input_token, &metadata, input_decimals).await?;
        
        // For new tokens, verify they meet basic security requirements
        if !self.verify_token_security(output_token).await? {
            return Err(SandoError::Strategy {
                kind: StrategyErrorKind::SecurityRisk,
                message: format!("Token {} failed security verification", output_token),
            });
        }
        
        // For token snipes, we use higher priority fees to increase chances of success
        let compute_budget_instruction = solana_sdk::compute_budget::ComputeBudgetInstruction::set_compute_unit_price(
            10000 // High priority fee for token snipes (10k micro-lamports)
        );
        
        // Set compute unit limit to ensure we have enough units for the transaction
        let compute_limit_instruction = solana_sdk::compute_budget::ComputeBudgetInstruction::set_compute_unit_limit(
            300_000 // Higher limit for token snipes, which can be complex
        );
        
        info!(
            input_token = %input_token,
            output_token = %output_token,
            amount = amount_in,
            "Fetching Jupiter quote for token snipe"
        );

        let quote_response = Jupiter::fetch_quote(&input_token, output_token, amount_in).await
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to fetch Jupiter quote for snipe: {}", e),
            })?;

        // For snipes, we use higher slippage tolerance since new tokens often have high volatility
        let slippage = 0.05; // 5% slippage for token snipes
        let out_amount = quote_response.out_amount.parse::<u64>()
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::InvalidParameters,
                message: format!("Failed to parse output amount: {}", e),
            })?;
        let min_out_amount = ((out_amount as f64) * (1.0 - slippage)) as u64;

        info!(
            expected_output = quote_response.out_amount.parse::<u64>().unwrap_or(0),
            min_output = min_out_amount,
            slippage_percent = slippage * 100.0,
            "Calculated minimum output amount for token snipe"
        );
        
        // Get swap instructions
        let swap_instructions = Jupiter::get_swap_instructions(
            quote_response,
            &self.executor.signer_pubkey(),
            min_out_amount
        ).await
            .map_err(|e| SandoError::Strategy {
                kind: StrategyErrorKind::ApiError,
                message: format!("Failed to get Jupiter swap instructions for snipe: {}", e),
            })?;
        
        // Combine all instructions
        let mut all_instructions = vec![compute_budget_instruction, compute_limit_instruction];
        all_instructions.extend(swap_instructions);
        
        // Create the transaction
        let message = Message::new(
            &all_instructions,
            Some(&self.executor.signer_pubkey()),
        );
        
        Ok(Transaction::new_unsigned(message))
    }
    
    /// Determine the best input token to use based on wallet balances
    async fn determine_best_input_token(&self) -> Result<String> {
        // Check balances for common tokens
        let stable_tokens = [
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", // USDC
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", // USDT
        ];
        
        let sol_pubkey = Pubkey::from_str("So11111111111111111111111111111111111111112").unwrap();
        
        // Check SOL balance first
        let sol_balance = self.executor.get_wallet_balance().await?;
        let min_sol_needed = 0.2 * 1_000_000_000 as f64; // 0.2 SOL in lamports
        
        // If we have enough SOL, use that (keeping some for gas)
        if sol_balance as f64 > min_sol_needed {
            return Ok("So11111111111111111111111111111111111111112".to_string()); // Native SOL
        }
        
        // Otherwise, try stable tokens
        for token in stable_tokens.iter() {
            if let Ok(token_pubkey) = Pubkey::from_str(token) {
                if let Ok(balance) = self.get_token_balance(&token_pubkey).await {
                    // If we have a balance, use this token
                    if balance > 0 {
                        return Ok(token.to_string());
                    }
                }
            }
        }
        
        // Default to SOL if no other token has a balance
        Ok("So11111111111111111111111111111111111111112".to_string())
    }
    
    /// Get token balance for a specific mint
    async fn get_token_balance(&self, mint: &Pubkey) -> Result<u64> {
        let owner = self.executor.signer_pubkey();
        let token_account = self.executor.get_associated_token_address(&owner, mint);
        
        match self.executor.rpc_client.get_token_account_balance(&token_account) {
            Ok(ui_amount) => Ok(ui_amount.amount.parse::<u64>().unwrap_or(0)),
            Err(_) => Ok(0), // Assume zero balance if error (likely account doesn't exist)
        }
    }
    
    /// Calculate the optimal amount to use for a token snipe
    async fn calculate_snipe_amount(&self, input_token: &str, metadata: &TokenSnipeMetadata, input_decimals: u8) -> Result<u64> {
        // Base investment on liquidity and token risk profile
        let liquidity = metadata.liquidity;
        let social_mentions = metadata.social_mentions as f64;
        
        // Calculate a risk score (0-1) where 1 is lowest risk
        let risk_score = match social_mentions {
            n if n > 1000.0 => 0.8, // High social mentions, lower risk
            n if n > 500.0 => 0.6,  // Moderate social mentions
            n if n > 100.0 => 0.4,  // Some social mentions
            _ => 0.2                // Few or no social mentions, high risk
        };
        
        // Base amount on liquidity (0.5-5% of available liquidity)
        let percent_of_liquidity = 0.005 + (risk_score * 0.045); // 0.5% to 5% based on risk
        let base_amount = liquidity * percent_of_liquidity;
        
        // Convert to token units
        let amount = (base_amount * 10_f64.powi(input_decimals as i32)) as u64;
        
        // Set min/max bounds based on token type
        let max_amount = match input_token {
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" => 2_000 * 10_u64.pow(6), // 2,000 USDC max
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB" => 2_000 * 10_u64.pow(6), // 2,000 USDT max
            "So11111111111111111111111111111111111111112" => 2 * 10_u64.pow(9),      // 2 SOL max
            _ => 1_000 * 10_u64.pow(input_decimals as u32),
        };
        
        let min_amount = match input_token {
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v" => 10 * 10_u64.pow(6),   // 10 USDC min
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB" => 10 * 10_u64.pow(6),   // 10 USDT min
            "So11111111111111111111111111111111111111112" => {
                // Fix the floating point multiplication with u64
                let decimal_factor = 10_u64.pow(9);
                (0.1 * decimal_factor as f64) as u64 // 0.1 SOL min
            },
            _ => 10 * 10_u64.pow(input_decimals as u32),
        };
        
        let final_amount = amount.clamp(min_amount, max_amount);
        
        info!(
            token = %input_token,
            risk_score = risk_score,
            liquidity = liquidity,
            social_mentions = social_mentions,
            percent_of_liquidity = percent_of_liquidity * 100.0,
            calculated_amount = amount,
            final_amount = final_amount,
            "Determined optimal amount for token snipe"
        );
        
        Ok(final_amount)
    }
    
    /// Verify token security - check for potential risks
    async fn verify_token_security(&self, token_mint: &str) -> Result<bool> {
        // Import the spl_token::id here
        use spl_token::id as spl_token_id;
        
        // In a real implementation, we would:
        // 1. Check token code for backdoors/honeypots
        // 2. Verify liquidity lock status
        // 3. Check token ownership concentration
        // 4. Look for mint authority being renounced
        // 5. Check for blocklist/allowlist functions
        
        // For this implementation, we'll just do some basic checks
        if let Ok(token_pubkey) = Pubkey::from_str(token_mint) {
            // Try to get token info from on-chain
            if let Ok(token_account) = self.executor.rpc_client.get_account(&token_pubkey) {
                // Ensure it's actually a token mint
                // Convert spl_token_id() to the expected Pubkey type
                let spl_token_program_id = Pubkey::from_str(spl_token_id().to_string().as_str()).unwrap_or_default();
                
                if token_account.owner == spl_token_program_id {
                    // Placeholder for real security checks
                    return Ok(true);
                } else {
                    warn!(
                        token_mint = %token_mint,
                        actual_owner = %token_account.owner,
                        expected_owner = %spl_token_program_id,
                        "Token mint is not owned by SPL Token Program"
                    );
                    return Ok(false);
                }
            }
        }
        
        // If we can't verify, assume it's not safe
        warn!(token_mint = %token_mint, "Unable to verify token security");
        Ok(false)
    }
}

#[async_trait]
impl StrategyExecutionService for StrategyExecutor {
    async fn execute_opportunity(&self, opportunity: &MevOpportunity) -> anyhow::Result<solana_sdk::signature::Signature> {
        // Convert SandoError to anyhow::Error
        self.execute_opportunity(opportunity).await.map_err(|e| anyhow::anyhow!("{}", e))
    }
} 