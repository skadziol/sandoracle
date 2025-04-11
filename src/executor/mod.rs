use crate::error::{Result, SandoError};
use crate::evaluator::MevOpportunity;
use solana_sdk::transaction::Transaction;
use solana_client::rpc_client::RpcClient;
use solana_sdk::signer::keypair::Keypair;
use solana_sdk::signer::Signer;
use std::sync::Arc;
use tracing::{info, error, warn};
use bs58;
use solana_sdk::instruction::Instruction;
use solana_sdk::message::Message;
use solana_sdk::pubkey::Pubkey;
use std::collections::HashMap;
use spl_associated_token_account::get_associated_token_address;
use std::str::FromStr;
use solana_sdk::native_token::LAMPORTS_PER_SOL;
use solana_client::rpc_config::{RpcSimulateTransactionConfig};
use spl_token::state::Mint;
use spl_token::id as spl_token_id;
use solana_sdk::account::Account;
use spl_token::state::Account as TokenAccount;
use spl_associated_token_account::solana_program::program_pack::Pack;

/// Represents the result of a transaction simulation
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub estimated_gas_cost: u64,
    pub estimated_profit_sol: f64,
    pub estimated_profit_usd: f64,
    pub is_profitable: bool,
    pub safety_checks_passed: bool,
    pub error: Option<String>,
    // Added fields for better simulation tracking
    pub token_balance_changes: HashMap<String, TokenBalanceChange>,
    pub instruction_logs: Vec<String>,
    pub compute_units_consumed: u64,
    pub accounts_referenced: Vec<String>,
}

/// Represents a token balance change in the simulation
#[derive(Debug, Clone, Default)]
pub struct TokenBalanceChange {
    pub mint: String,
    pub ui_amount_change: f64,
    pub ui_amount_before: f64,
    pub ui_amount_after: f64,
}

/// Handles the simulation and execution of MEV transactions
#[derive(Clone)]
pub struct TransactionExecutor {
    rpc_client: Arc<RpcClient>,
    signer: Arc<Keypair>,
    simulation_mode: bool, // Added simulation mode flag
}

impl std::fmt::Debug for TransactionExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransactionExecutor")
            .field("rpc_client", &"<RpcClient>")
            // Add other fields here
            .finish()
    }
}

// --- Structs for Arbitrage Strategy ---

/// Represents a single step in an arbitrage path
#[derive(Debug, Clone)] // Add derives as needed
pub struct ArbitrageStep {
    pub dex_program_id: Pubkey,
    pub input_token_mint: Pubkey,
    pub output_token_mint: Pubkey,
    pub input_amount: u64,
    pub min_output_amount: u64, // For slippage control
    // Add any other DEX-specific parameters (e.g., pool addresses)
}

/// Represents a full arbitrage path
#[derive(Debug, Clone)]
pub struct ArbitragePath {
    pub steps: Vec<ArbitrageStep>,
}

impl TransactionExecutor {
    /// Creates a new TransactionExecutor
    pub fn new(
        rpc_url: &str, 
        wallet_private_key_bs58: &str, 
        simulation_mode: bool, // Add flag to constructor
    ) -> Result<Self> {
        let rpc_client = Arc::new(RpcClient::new(rpc_url.to_string()));
        
        // Decode the base58 private key string
        let private_key_bytes = bs58::decode(wallet_private_key_bs58)
            .into_vec()
            .map_err(|e| SandoError::ConfigError(format!("Invalid base58 private key: {}", e)))?;
            
        // Create Keypair from decoded bytes
        let signer = Keypair::from_bytes(&private_key_bytes)
            .map_err(|e| SandoError::ConfigError(format!("Failed to create keypair from bytes: {}", e)))?;
        
        info!(signer_pubkey = %signer.pubkey(), simulation_mode = simulation_mode, "TransactionExecutor initialized");

        Ok(Self { 
            rpc_client,
            signer: Arc::new(signer),
            simulation_mode, // Store the flag
        })
    }

    /// Returns the public key of the signer wallet.
    pub fn signer_pubkey(&self) -> Pubkey {
        self.signer.pubkey()
    }

    /// Fetches the current balance of the signer wallet.
    pub async fn get_wallet_balance(&self) -> Result<u64> {
        self.rpc_client
            .get_balance(&self.signer_pubkey())
            .map_err(|e| SandoError::SolanaRpc(format!("Failed to get wallet balance: {}", e)))
    }

    /// Gets the associated token account address - static method using token program's functions
    fn get_associated_token_address(&self, owner: &Pubkey, mint: &Pubkey) -> Pubkey {
        // Use a more compatible approach to get the associated token address
        let owner_key = owner.to_string();
        let mint_key = mint.to_string();
        
        // Convert back to SDK pubkey since we're working with SDK APIs
        if let (Ok(owner_pubkey), Ok(mint_pubkey)) = (Pubkey::from_str(&owner_key), Pubkey::from_str(&mint_key)) {
            // Use the SDK version of associated token program
            let program_id = Pubkey::from_str("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL").unwrap_or_default();
            
            // Use the token program ID
            let token_program_id = Pubkey::from_str("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA").unwrap_or_default();
            
            // Hard-code the derivation logic for associated token accounts
            let seeds = &[
                owner_pubkey.as_ref(),
                token_program_id.as_ref(),
                mint_pubkey.as_ref(),
            ];
            
            // Make sure both program_id and return value are solana_sdk::pubkey::Pubkey
            let (associated_token_address, _bump_seed) = solana_sdk::pubkey::Pubkey::find_program_address(
                seeds, 
                &program_id
            );
            
            return associated_token_address;
        }
        
        // Fallback in case of error
        Pubkey::default()
    }

    /// Simulates a potential MEV transaction before execution
    /// 
    /// # Arguments
    /// * `opportunity` - The MevOpportunity to simulate.
    /// * `transaction` - The unsigned transaction to simulate.
    /// 
    /// # Returns
    /// A `Result` containing the `SimulationResult`.
    pub async fn simulate_transaction(
        &self,
        opportunity: &MevOpportunity,
        transaction: &Transaction,
    ) -> Result<SimulationResult> {
        info!(strategy = ?opportunity.strategy, profit = opportunity.estimated_profit, "Simulating transaction...");
        
        // --- 1. Get Pre-Transaction State ---
        let pre_balances = self.get_token_balances(&opportunity.involved_tokens).await?;
        
        // --- 2. Simulate Transaction ---
        let config = RpcSimulateTransactionConfig {
            sig_verify: false,
            replace_recent_blockhash: true,
            commitment: None,
            encoding: None,
            accounts: None, // We'll handle account tracking separately
            min_context_slot: None,
            inner_instructions: true,
        };

        let simulation_result = self.rpc_client
            .simulate_transaction_with_config(transaction, config)
            .map_err(|e| SandoError::Simulation(format!("RPC simulation failed: {}", e)))?;

        if let Some(err) = simulation_result.value.err {
            error!(error = ?err, "Transaction simulation returned error");
            return Ok(SimulationResult {
                estimated_gas_cost: 0,
                estimated_profit_sol: 0.0,
                estimated_profit_usd: 0.0,
                is_profitable: false,
                safety_checks_passed: false,
                error: Some(format!("Simulation error: {:?}", err)),
                token_balance_changes: HashMap::new(),
                instruction_logs: Vec::new(),
                compute_units_consumed: 0,
                accounts_referenced: Vec::new(),
            });
        }

        // --- 3. Extract Simulation Data ---
        let compute_units_consumed = simulation_result.value.units_consumed.unwrap_or(0);
        let estimated_gas_cost = compute_units_consumed * 5000; // 5000 lamports per CU
        
        let logs = simulation_result.value.logs.unwrap_or_default();
        
        // Process accounts if available
        let accounts_referenced: Vec<String> = Vec::new();
        let post_balances = pre_balances.clone();
        
        if let Some(accounts) = &simulation_result.value.accounts {
            for (_i, acc_opt) in accounts.iter().enumerate() {
                if let Some(_acc_data) = acc_opt {
                    // Process account data if needed
                    // This would update post_balances based on account state
                }
            }
        }
        
        // --- 4. Extract Token Balance Changes ---
        // Here we'd parse balance changes from logs or post-transaction state
        // For now, we'll simulate these changes for testing
        let token_balance_changes = self.calculate_token_balance_changes(pre_balances, post_balances)?;
        
        // --- 5. Calculate Profit ---
        let (profit_sol, profit_usd) = self.calculate_profit(&token_balance_changes, estimated_gas_cost, opportunity).await?;
        
        // --- 6. Perform Safety Checks ---
        let accounts: Vec<Account> = Vec::new(); // We'd extract accounts from simulation data
        let instruction_count = transaction.message().instructions.len() as u64;
        let (safety_checks_passed, safety_error) = self.perform_safety_checks(
            opportunity, 
            &token_balance_changes, 
            &accounts, 
            instruction_count
        ).await?;
        
        // --- 7. Build Final Simulation Result ---
        let is_profitable = profit_sol > 0.0 && (estimated_gas_cost as f64 / LAMPORTS_PER_SOL as f64) < profit_sol;
        
        let result = SimulationResult {
            estimated_gas_cost,
            estimated_profit_sol: profit_sol,
            estimated_profit_usd: profit_usd,
            is_profitable,
            safety_checks_passed,
            error: safety_error,
            token_balance_changes,
            instruction_logs: logs,
            compute_units_consumed,
            accounts_referenced: accounts_referenced,
        };
        
        info!(
            profit_sol = result.estimated_profit_sol,
            profit_usd = result.estimated_profit_usd,
            gas_cost = result.estimated_gas_cost,
            is_profitable = result.is_profitable,
            safety_passed = result.safety_checks_passed,
            "Simulation completed"
        );
        
        Ok(result)
    }

    /// Calculates token balance changes from pre and post simulation states
    fn calculate_token_balance_changes(
        &self,
        pre_balances: HashMap<String, TokenBalanceChange>,
        post_balances: HashMap<String, TokenBalanceChange>,
    ) -> Result<HashMap<String, TokenBalanceChange>> {
        let mut changes = HashMap::new();
        
        // Process all tokens from pre-balances
        for (token, pre_balance) in pre_balances.iter() {
            let mut change = pre_balance.clone();
            
            // If we have post-balance data, calculate the difference
            if let Some(post_balance) = post_balances.get(token) {
                change.ui_amount_after = post_balance.ui_amount_after;
                change.ui_amount_change = post_balance.ui_amount_after - pre_balance.ui_amount_before;
            }
            
            changes.insert(token.clone(), change);
        }
        
        // Add any tokens that only exist in post-balances
        for (token, post_balance) in post_balances.iter() {
            if !pre_balances.contains_key(token) {
                let mut change = post_balance.clone();
                change.ui_amount_change = post_balance.ui_amount_after; // Assuming started at 0
                changes.insert(token.clone(), change);
            }
        }
        
        Ok(changes)
    }

    /// Gets token balances for a list of tokens
    async fn get_token_balances(&self, tokens: &[String]) -> Result<HashMap<String, TokenBalanceChange>> {
        let mut balances = HashMap::new();
        let owner = self.signer_pubkey();
        
        for token_str in tokens {
            if let Ok(mint) = Pubkey::from_str(token_str) {
                let token_account = self.get_associated_token_address(&owner, &mint);
                
                match self.rpc_client.get_token_account_balance(&token_account) {
                    Ok(balance) => {
                        let balance_change = TokenBalanceChange {
                            mint: token_str.clone(),
                            ui_amount_change: 0.0, // Will be calculated after simulation
                            ui_amount_before: balance.ui_amount.unwrap_or(0.0),
                            ui_amount_after: balance.ui_amount.unwrap_or(0.0), // Same as before, will be updated
                        };
                        balances.insert(token_str.clone(), balance_change);
                    },
                    Err(e) => {
                        // Token account may not exist yet, log and continue
                        warn!(token = token_str, error = %e, "Token account not found or other error");
                        let balance_change = TokenBalanceChange {
                            mint: token_str.clone(),
                            ui_amount_change: 0.0,
                            ui_amount_before: 0.0,
                            ui_amount_after: 0.0,
                        };
                        balances.insert(token_str.clone(), balance_change);
                    }
                }
            } else {
                warn!(token = token_str, "Invalid token mint address");
            }
        }
        
        // Add native SOL balance
        let sol_balance = self.get_wallet_balance().await?;
        let sol_balance_change = TokenBalanceChange {
            mint: "SOL".to_string(),
            ui_amount_change: 0.0,
            ui_amount_before: sol_balance as f64 / LAMPORTS_PER_SOL as f64,
            ui_amount_after: sol_balance as f64 / LAMPORTS_PER_SOL as f64,
        };
        balances.insert("SOL".to_string(), sol_balance_change);
        
        Ok(balances)
    }

    /// Gets the decimals of a token mint
    async fn get_mint_decimals(&self, mint: &Pubkey) -> Result<u8> {
        let account_info = self.rpc_client.get_account(mint)
            .map_err(|e| SandoError::SolanaRpc(format!("Failed to get mint account: {}", e)))?;
            
        let mint_info = Mint::unpack_from_slice(&account_info.data)
            .map_err(|e| SandoError::ConfigError(format!("Failed to unpack mint data: {}", e)))?;
            
        Ok(mint_info.decimals)
    }

    /// Calculates profit from token changes and gas cost
    async fn calculate_profit(
        &self,
        token_changes: &HashMap<String, TokenBalanceChange>,
        gas_cost: u64,
        _opportunity: &MevOpportunity, // Unused parameter
    ) -> Result<(f64, f64)> {
        let mut total_profit_sol = 0.0;
        
        // Sum up all token changes in SOL value
        for (token, change) in token_changes.iter() {
            if token == "SOL" {
                total_profit_sol += change.ui_amount_change;
            } else {
                // Convert token to SOL value
                let token_price_sol = self.get_token_price_in_sol(token).await?;
                total_profit_sol += change.ui_amount_change * token_price_sol;
            }
        }
        
        // Subtract gas cost
        total_profit_sol -= gas_cost as f64 / LAMPORTS_PER_SOL as f64;
        
        // Convert SOL profit to USD
        let sol_price_usd = self.get_sol_price_usd().await?;
        let total_profit_usd = total_profit_sol * sol_price_usd;
        
        Ok((total_profit_sol, total_profit_usd))
    }

    /// Gets token price in SOL (placeholder implementation)
    async fn get_token_price_in_sol(&self, mint: &str) -> Result<f64> {
        // In a real implementation, this would fetch token prices from an oracle
        // For now, we'll use dummy values for demonstration
        match mint {
            "USDC" => Ok(0.04), // 1 USDC = 0.04 SOL (example value)
            "BONK" => Ok(0.0000001), // Just an example rate
            _ => Ok(0.01), // Default fallback
        }
    }

    /// Gets SOL price in USD (placeholder implementation)
    async fn get_sol_price_usd(&self) -> Result<f64> {
        // In a real implementation, this would fetch SOL price from an oracle
        Ok(25.0) // Example: 1 SOL = $25 USD
    }

    /// Performs safety checks on the transaction
    async fn perform_safety_checks(
        &self,
        opportunity: &MevOpportunity,
        token_changes: &HashMap<String, TokenBalanceChange>,
        _accounts: &[Account],
        instruction_count: u64,
    ) -> Result<(bool, Option<String>)> {
        // --- 1. Check token balance changes ---
        for (token, change) in token_changes.iter() {
            // Ensure we're only spending allowed output tokens
            if change.ui_amount_change < 0.0 && !opportunity.allowed_output_tokens.contains(token) {
                return Ok((false, Some(format!("Unauthorized token spend: {}", token))));
            }
            
            // Check for excessive or unexpected token changes
            if change.ui_amount_change.abs() > opportunity.required_capital * 2.0 {
                return Ok((false, Some(format!("Excessive token change: {}", token))));
            }
        }
        
        // --- 2. Check instruction count ---
        if instruction_count > opportunity.max_instructions {
            return Ok((false, Some(format!(
                "Instruction count ({}) exceeds maximum allowed ({})",
                instruction_count, opportunity.max_instructions
            ))));
        }
        
        // --- 3. Check for unauthorized program calls ---
        // This would need to parse the accounts referenced in the transaction
        // or analyze the simulation logs for program invocations
        
        // --- 4. Check for suspicious account references ---
        // In a real implementation, this would analyze accounts referenced
        // in the transaction against a whitelist or known patterns
        
        // All checks passed
        Ok((true, None))
    }

    // Updated execute_transaction
    pub async fn execute_transaction(&self, mut transaction: Transaction) -> Result<solana_sdk::signature::Signature> {
        info!("Attempting to execute transaction...");

        // --- Check Simulation Mode --- 
        if self.simulation_mode {
            warn!("SIMULATION MODE ACTIVE: Transaction will not be sent.");
            // Optionally sign and log, but don't send
             let recent_blockhash = self.rpc_client
                .get_latest_blockhash()
                .map_err(|e| SandoError::SolanaRpc(format!("Failed to get recent blockhash in sim mode: {}", e)))?;
            transaction.sign(&[self.signer.as_ref()], recent_blockhash);
            let sim_signature = transaction.signatures[0];
            info!(simulated_signature = %sim_signature, "Signed transaction in simulation mode.");
            // TODO: Add detailed execution tracking log for simulation
            return Ok(sim_signature); // Return the would-be signature
        }

        // --- Actual Execution Logic (if not in simulation mode) ---
        info!("Executing transaction on-chain...");

        // --- 2. Sign Transaction ---
        let recent_blockhash = self.rpc_client
            .get_latest_blockhash() // Removed .await
            .map_err(|e| SandoError::SolanaRpc(format!("Failed to get recent blockhash: {}", e)))?;
        
        transaction.sign(&[self.signer.as_ref()], recent_blockhash);
        info!(signature = %transaction.signatures[0], "Transaction signed");

        // --- 3. Send Transaction with Retry Logic ---
        let max_retries = 5;
        let mut attempt = 0;
        let base_delay = tokio::time::Duration::from_millis(500);

        loop {
            attempt += 1;
            info!(attempt = attempt, max_retries = max_retries, "Sending transaction...");

            match self.rpc_client.send_and_confirm_transaction(&transaction) {
                Ok(signature) => {
                    info!(signature = %signature, "Transaction confirmed successfully!");
                    // TODO: Add detailed execution tracking (log to db/file)
                    return Ok(signature);
                }
                Err(e) => {
                    let sando_error = SandoError::SolanaRpc(e.to_string()); // Convert client error
                    error!(attempt = attempt, error = %sando_error, "Send/confirm transaction failed");
                    
                    // Check if retryable and if max retries exceeded
                    if attempt >= max_retries || !sando_error.is_retryable() {
                        error!("Max retries reached or error not retryable. Giving up.");
                        // TODO: Add detailed execution tracking for failure
                        return Err(sando_error);
                    }
                    
                    // Calculate exponential backoff delay
                    let delay = base_delay * 2u32.pow(attempt - 1);
                    warn!(delay = ?delay, "Retrying transaction after delay...");
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// Builds an unsigned transaction for a given arbitrage path
    pub fn build_arbitrage_transaction(
        &self,
        arbitrage_path: &ArbitragePath,
    ) -> Result<Transaction> {
        info!(steps = arbitrage_path.steps.len(), "Building arbitrage transaction...");
        
        let mut instructions: Vec<Instruction> = Vec::new();

        for (index, step) in arbitrage_path.steps.iter().enumerate() {
            info!(step = index + 1, dex = %step.dex_program_id, input = %step.input_token_mint, output = %step.output_token_mint, "Processing arbitrage step");
            
            // TODO: Implement logic to create specific swap instructions based on step.dex_program_id
            // This will require specific knowledge of each DEX's program interface.
            // Example structure:
            // match step.dex_program_id {
            //     ORCA_SWAP_PROGRAM_ID => {
            //         let ix = create_orca_swap_instruction(
            //             &self.signer.pubkey(), // Payer/user account
            //             step.input_token_mint,
            //             step.output_token_mint,
            //             step.input_amount,
            //             step.min_output_amount,
            //             // ... other necessary accounts (pools, authority, token accounts) ... 
            //         )?;
            //         instructions.push(ix);
            //     }
            //     RAYDIUM_PROGRAM_ID => { /* create Raydium instruction */ }
            //     JUPITER_PROGRAM_ID => { /* create Jupiter instruction (might involve multiple instructions) */ }
            //     _ => {
            //         error!(unknown_dex = %step.dex_program_id, "Unsupported DEX program ID for arbitrage step");
            //         return Err(SandoError::Strategy {
            //             kind: crate::error::StrategyErrorKind::InvalidParameters,
            //             message: format!("Unsupported DEX: {}", step.dex_program_id)
            //         });
            //     }
            // }
            
            // Placeholder: Add a dummy instruction for now to allow compilation
            warn!("Using dummy instruction for arbitrage step {}", index + 1);
            let dummy_recipient = Pubkey::new_unique();
            instructions.push(
                solana_sdk::system_instruction::transfer(
                    &self.signer.pubkey(),
                    &dummy_recipient,
                    1, // 1 lamport
                )
            );
        }

        if instructions.is_empty() {
            return Err(SandoError::Strategy {
                kind: crate::error::StrategyErrorKind::InvalidParameters,
                message: "No instructions generated for arbitrage path".to_string(),
            });
        }

        // Assemble the transaction
        let message = Message::new(&instructions, Some(&self.signer.pubkey()));
        // Note: The blockhash will be added later during signing in execute_transaction
        let transaction = Transaction::new_unsigned(message);

        info!("Arbitrage transaction built successfully (using placeholders)");
        Ok(transaction)
    }
}

// Add basic unit tests for the executor structure
#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::{MevOpportunity, RiskLevel, Strategy};
    use solana_sdk::message::Message;
    use solana_sdk::pubkey::Pubkey;
    use solana_sdk::signature::{Keypair, Signer};
    use std::str::FromStr;

    // Helper to get a test keypair
    fn get_test_keypair() -> (Keypair, String) {
        let kp = Keypair::new();
        let b58 = bs58::encode(kp.to_bytes()).into_string();
        (kp, b58)
    }

    // Helper to create a test opportunity
    fn create_test_opportunity() -> MevOpportunity {
        MevOpportunity {
            strategy: Strategy::Arbitrage,
            estimated_profit: 0.1,
            risk_level: RiskLevel::Low,
            involved_tokens: vec![
                solana_sdk::native_token::id().to_string(),
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string(), // USDC
            ],
            allowed_output_tokens: vec![
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string(), // USDC
            ],
            allowed_programs: vec![
                solana_sdk::system_program::id(),
                Pubkey::from_str("9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin").unwrap(), // Serum
            ],
            max_instructions: 10,
            timeout: std::time::Duration::from_secs(30),
        }
    }

    #[test]
    fn test_executor_creation_with_wallet() {
        let (_kp, b58_key) = get_test_keypair();
        let rpc_url = "https://api.devnet.solana.com"; 
        // Test both simulation modes
        let executor_sim_true = TransactionExecutor::new(rpc_url, &b58_key, true);
        assert!(executor_sim_true.is_ok());
        assert_eq!(executor_sim_true.unwrap().simulation_mode, true);

        let executor_sim_false = TransactionExecutor::new(rpc_url, &b58_key, false);
        assert!(executor_sim_false.is_ok());
        assert_eq!(executor_sim_false.unwrap().simulation_mode, false);
    }

    #[test]
    fn test_executor_creation_invalid_key() {
        let rpc_url = "https://api.devnet.solana.com"; 
        let executor_result = TransactionExecutor::new(rpc_url, "invalid-base58-key", false);
        assert!(executor_result.is_err());
        assert!(matches!(executor_result.unwrap_err(), SandoError::ConfigError(_)));
    }

    #[test]
    fn test_build_arbitrage_transaction_structure() {
        let (_kp, b58_key) = get_test_keypair();
        let rpc_url = "https://api.devnet.solana.com"; 
        let executor = TransactionExecutor::new(rpc_url, &b58_key, false).unwrap();

        // Define a dummy arbitrage path
        let dummy_dex = Pubkey::new_unique();
        let token_a = Pubkey::new_unique();
        let token_b = Pubkey::new_unique();
        let token_c = Pubkey::new_unique();

        let path = ArbitragePath {
            steps: vec![
                ArbitrageStep {
                    dex_program_id: dummy_dex,
                    input_token_mint: token_a,
                    output_token_mint: token_b,
                    input_amount: 1000,
                    min_output_amount: 990,
                },
                ArbitrageStep {
                    dex_program_id: dummy_dex,
                    input_token_mint: token_b,
                    output_token_mint: token_c,
                    input_amount: 990, 
                    min_output_amount: 980,
                },
            ]
        };

        let result = executor.build_arbitrage_transaction(&path);
        assert!(result.is_ok());
        let tx = result.unwrap();
        assert_eq!(tx.message.instructions.len(), path.steps.len()); // Should have one instruction per step (currently dummy)
        assert_eq!(tx.message.account_keys[0], executor.signer.pubkey()); // First key should be payer
    }

    #[tokio::test]
    async fn test_execute_transaction_simulation_mode() {
        // Test that execute_transaction returns Ok without sending in simulation mode
        let (_kp, b58_key) = get_test_keypair();
        let rpc_url = "https://api.devnet.solana.com"; // Use devnet for blockhash
        let executor = TransactionExecutor::new(rpc_url, &b58_key, true).unwrap(); // Simulation mode = true

        // Create a simple dummy transaction
        let recipient = Pubkey::new_unique();
        let instruction = solana_sdk::system_instruction::transfer(
            &executor.signer.pubkey(),
            &recipient,
            1, // lamports
        );
        let message = Message::new(&[instruction], Some(&executor.signer.pubkey()));
        // Blockhash added during execute_transaction
        let tx = Transaction::new_unsigned(message);

        // Execute in simulation mode
        let result = executor.execute_transaction(tx).await;

        // Should return Ok with a signature, but no transaction sent
        assert!(result.is_ok(), "execute_transaction in sim mode failed: {:?}", result.err());
        let signature = result.unwrap();
        // Check if it looks like a valid signature (not default)
        assert_ne!(signature, solana_sdk::signature::Signature::default()); 
        // We can't easily verify it wasn't sent without more complex mocking or querying the network
        // but the function should return Ok without panicking or hitting the network send.
    }

    #[tokio::test]
    async fn test_simulation_success() {
        let (_kp, b58_key) = get_test_keypair();
        let rpc_url = "https://api.devnet.solana.com";
        let executor = TransactionExecutor::new(rpc_url, &b58_key, false).unwrap();

        // Create a simple SOL transfer transaction
        let recipient = Pubkey::new_unique();
        let instruction = solana_sdk::system_instruction::transfer(
            &executor.signer.pubkey(),
            &recipient,
            1_000_000, // 0.001 SOL
        );
        let message = Message::new(&[instruction], Some(&executor.signer.pubkey()));
        let transaction = Transaction::new_unsigned(message);

        let opportunity = create_test_opportunity();
        let result = executor.simulate_transaction(&opportunity, &transaction).await;

        assert!(result.is_ok(), "Simulation failed: {:?}", result.err());
        let sim_result = result.unwrap();

        // Basic checks
        assert!(sim_result.compute_units_consumed > 0);
        assert!(!sim_result.instruction_logs.is_empty());
        assert!(sim_result.token_balance_changes.contains_key(&solana_sdk::native_token::id().to_string()));
        assert!(sim_result.safety_checks_passed);
    }

    #[tokio::test]
    async fn test_simulation_safety_checks() {
        let (_kp, b58_key) = get_test_keypair();
        let rpc_url = "https://api.devnet.solana.com";
        let executor = TransactionExecutor::new(rpc_url, &b58_key, false).unwrap();

        // Create a transaction that should fail safety checks
        let unauthorized_program = Pubkey::new_unique();
        let instruction = Instruction::new_with_bytes(
            unauthorized_program,
            &[0],
            vec![],
        );
        let message = Message::new(&[instruction], Some(&executor.signer.pubkey()));
        let transaction = Transaction::new_unsigned(message);

        let opportunity = create_test_opportunity();
        let result = executor.simulate_transaction(&opportunity, &transaction).await;

        assert!(result.is_ok());
        let sim_result = result.unwrap();

        // Should fail safety checks due to unauthorized program
        assert!(!sim_result.safety_checks_passed);
        assert!(sim_result.error.unwrap().contains("Unauthorized program"));
    }

    #[tokio::test]
    async fn test_profit_calculation() {
        let (_kp, b58_key) = get_test_keypair();
        let rpc_url = "https://api.devnet.solana.com";
        let executor = TransactionExecutor::new(rpc_url, &b58_key, false).unwrap();

        // Create a test token balance change
        let mut token_changes = HashMap::new();
        token_changes.insert(
            solana_sdk::native_token::id().to_string(),
            TokenBalanceChange {
                mint: solana_sdk::native_token::id().to_string(),
                ui_amount_change: 0.1,
                ui_amount_before: 1_000_000_000, // 1 SOL
                ui_amount_after: 1_100_000_000, // 1.1 SOL
            },
        );

        let opportunity = create_test_opportunity();
        let gas_cost = 5_000_000; // 0.005 SOL

        let (profit_sol, profit_usd) = executor.calculate_profit(
            &token_changes,
            gas_cost,
            &opportunity,
        ).await.unwrap();

        // With 0.1 SOL gain and 0.005 SOL gas cost
        assert!(profit_sol > 0.0);
        // With placeholder SOL price of 100 USD
        assert!(profit_usd > 0.0);
    }
} 