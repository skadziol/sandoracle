// Executor unit tests

use sandoracle::{
    executor::{TransactionExecutor, ArbitragePath, ArbitrageStep},
    evaluator::{MevOpportunity, MevStrategy, RiskLevel},
    error::SandoError,
};
use solana_sdk::{
    message::Message, 
    pubkey::Pubkey, 
    signature::{Keypair, Signer}, 
    transaction::Transaction,
    instruction::Instruction,
};
use std::{str::FromStr, collections::HashMap};
use sandoracle::executor::TokenBalanceChange;

// Helper to get a test keypair
fn get_test_keypair() -> (Keypair, String) {
    let kp = Keypair::new();
    let b58 = bs58::encode(kp.to_bytes()).into_string();
    (kp, b58)
}

// Helper to create a test opportunity
fn create_test_opportunity() -> MevOpportunity {
    MevOpportunity {
        strategy: MevStrategy::Arbitrage,
        estimated_profit: 0.1,
        confidence: 0.9,
        risk_level: RiskLevel::Low,
        required_capital: 1.0,
        execution_time: 1000,
        metadata: serde_json::json!({}),
        score: None,
        decision: None,
        involved_tokens: vec![
            "So11111111111111111111111111111111111111112".to_string(), // SOL
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string(), // USDC
        ],
        allowed_output_tokens: vec![
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string(), // USDC
        ],
        allowed_programs: vec![
            solana_sdk::system_program::id().to_string(),
            "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin".to_string(), // Serum
        ],
        max_instructions: 10,
    }
}

#[test]
fn test_executor_creation_with_wallet() {
    let (_kp, b58_key) = get_test_keypair();
    let rpc_url = "https://api.devnet.solana.com"; 
    // Test both simulation modes
    let executor_sim_true = TransactionExecutor::new(rpc_url, &b58_key, true);
    assert!(executor_sim_true.is_ok());
    let executor = executor_sim_true.unwrap();
    assert!(executor.simulation_mode());

    let executor_sim_false = TransactionExecutor::new(rpc_url, &b58_key, false);
    assert!(executor_sim_false.is_ok());
    let executor = executor_sim_false.unwrap();
    assert!(!executor.simulation_mode());
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
                pool_address: None,
                additional_accounts: vec![],
            },
            ArbitrageStep {
                dex_program_id: dummy_dex,
                input_token_mint: token_b,
                output_token_mint: token_c,
                input_amount: 990, 
                min_output_amount: 980,
                pool_address: None,
                additional_accounts: vec![],
            },
        ]
    };

    let result = executor.build_arbitrage_transaction(&path);
    assert!(result.is_ok());
    let tx = result.unwrap();
    assert_eq!(tx.message.instructions.len(), path.steps.len()); // Should have one instruction per step (currently dummy)
    assert_eq!(tx.message.account_keys[0], executor.signer_pubkey()); // First key should be payer
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
        &executor.signer_pubkey(),
        &recipient,
        1, // lamports
    );
    let message = Message::new(&[instruction], Some(&executor.signer_pubkey()));
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
        &executor.signer_pubkey(),
        &recipient,
        1_000_000, // 0.001 SOL
    );
    let message = Message::new(&[instruction], Some(&executor.signer_pubkey()));
    let transaction = Transaction::new_unsigned(message);

    let opportunity = create_test_opportunity();
    let result = executor.simulate_transaction(&opportunity, &transaction).await;

    assert!(result.is_ok(), "Simulation failed: {:?}", result.err());
    let sim_result = result.unwrap();

    // Basic checks
    assert!(sim_result.compute_units_consumed > 0);
    assert!(!sim_result.instruction_logs.is_empty());
    assert!(sim_result.token_balance_changes.contains_key(
        &"So11111111111111111111111111111111111111112".to_string()
    ));
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
    let message = Message::new(&[instruction], Some(&executor.signer_pubkey()));
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
        "So11111111111111111111111111111111111111112".to_string(),
        TokenBalanceChange {
            mint: "So11111111111111111111111111111111111111112".to_string(),
            ui_amount_change: 0.1,
            ui_amount_before: 1_000_000_000.0, // 1 SOL
            ui_amount_after: 1_100_000_000.0, // 1.1 SOL
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