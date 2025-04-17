use crate::executor::{TransactionExecutor, strategies::StrategyExecutor};
use crate::executor::strategies::{ArbitrageMetadata, SandwichMetadata, TokenSnipeMetadata};
use crate::evaluator::{MevOpportunity, MevStrategy, RiskLevel};
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::{Keypair, Signer};
use std::str::FromStr;
use serde_json::json;
use std::sync::Arc;

// Helper to get a test keypair
fn get_test_keypair() -> (Keypair, String) {
    let kp = Keypair::new();
    let b58 = bs58::encode(kp.to_bytes()).into_string();
    (kp, b58)
}

// Helper to create a test RPC URL - using devnet for tests
fn get_test_rpc_url() -> String {
    "https://api.devnet.solana.com".to_string()
}

// Helper to create an arbitrage opportunity
fn create_arbitrage_opportunity() -> MevOpportunity {
    // Example token addresses (using mainnet tokens for realism)
    let sol = "So11111111111111111111111111111111111111112"; // SOL
    let usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"; // USDC
    let usdt = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"; // USDT
    
    // DEX program IDs (using real program IDs for realism)
    let raydium = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"; // Raydium
    let orca = "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP"; // Orca
    
    // Create metadata for the arbitrage opportunity
    let metadata = ArbitrageMetadata {
        token_path: vec![sol.to_string(), usdc.to_string(), usdt.to_string(), sol.to_string()],
        prices: vec![100.0, 1.0, 1.0, 101.0],
        liquidity: vec![100000.0, 200000.0, 200000.0, 100000.0],
        price_impacts: vec![0.001, 0.001, 0.001, 0.001],
        dexes: vec![raydium.to_string(), orca.to_string(), raydium.to_string()],
    };
    
    MevOpportunity {
        strategy: MevStrategy::Arbitrage,
        estimated_profit: 1.0, // 1 SOL profit
        confidence: 0.95,
        risk_level: RiskLevel::Low,
        required_capital: 100.0, // 100 SOL
        execution_time: 500, // 500ms
        metadata: serde_json::to_value(metadata).unwrap(),
        score: Some(0.85),
        decision: None,
        involved_tokens: vec![
            sol.to_string(),
            usdc.to_string(),
            usdt.to_string(),
        ],
        allowed_output_tokens: vec![
            sol.to_string(),
            usdc.to_string(),
            usdt.to_string(),
        ],
        allowed_programs: vec![
            raydium.to_string(),
            orca.to_string(),
            solana_sdk::system_program::id().to_string(),
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA".to_string(), // SPL Token
        ],
        max_instructions: 10,
    }
}

// Helper to create a sandwich opportunity
fn create_sandwich_opportunity() -> MevOpportunity {
    // Example token addresses
    let sol = "So11111111111111111111111111111111111111112"; // SOL
    let usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"; // USDC
    
    // DEX program ID
    let raydium = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"; // Raydium
    
    // Create metadata for the sandwich opportunity
    let metadata = SandwichMetadata {
        target_tx_hash: "5xhJHTVXQseaXb5FVqD4J3jKxEvz7GXHCDGkz9JCYAXxESktX45KuNbYH3XEPmJFbgSm5Px4JWwf8PeCqY4dKdXd".to_string(),
        token_pair: (sol.to_string(), usdc.to_string()),
        target_tx_size: 50.0, // 50 SOL
        front_run_amount: 10.0, // 10 SOL
        back_run_amount: 10.0, // 10 SOL
        front_run_impact: 0.005, // 0.5%
        back_run_impact: 0.005, // 0.5%
        dex: raydium.to_string(),
        gas_cost: 0.001, // 0.001 SOL
    };
    
    MevOpportunity {
        strategy: MevStrategy::Sandwich,
        estimated_profit: 0.5, // 0.5 SOL profit
        confidence: 0.9,
        risk_level: RiskLevel::Medium,
        required_capital: 20.0, // 20 SOL
        execution_time: 1000, // 1000ms
        metadata: serde_json::to_value(metadata).unwrap(),
        score: Some(0.75),
        decision: None,
        involved_tokens: vec![
            sol.to_string(),
            usdc.to_string(),
        ],
        allowed_output_tokens: vec![
            sol.to_string(),
            usdc.to_string(),
        ],
        allowed_programs: vec![
            raydium.to_string(),
            solana_sdk::system_program::id().to_string(),
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA".to_string(), // SPL Token
        ],
        max_instructions: 8,
    }
}

// Helper to create a token snipe opportunity
fn create_token_snipe_opportunity() -> MevOpportunity {
    // Example token addresses
    let sol = "So11111111111111111111111111111111111111112"; // SOL
    let new_token = Pubkey::new_unique().to_string(); // Newly launched token
    
    // DEX program ID
    let raydium = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"; // Raydium
    
    // Create metadata for the token snipe opportunity
    let metadata = TokenSnipeMetadata {
        token_address: new_token.clone(),
        token_name: "NewToken".to_string(),
        initial_price: 0.001, // 0.001 SOL
        social_mentions: 500,
        launch_time: chrono::Utc::now().timestamp(),
        volume: 10000.0,
        liquidity: 50000.0,
        creator: Pubkey::new_unique().to_string(),
        gas_cost: 0.002, // 0.002 SOL
        dex: raydium.to_string(),
    };
    
    MevOpportunity {
        strategy: MevStrategy::TokenSnipe,
        estimated_profit: 5.0, // 5 SOL profit
        confidence: 0.8,
        risk_level: RiskLevel::High,
        required_capital: 10.0, // 10 SOL
        execution_time: 300, // 300ms
        metadata: serde_json::to_value(metadata).unwrap(),
        score: Some(0.7),
        decision: None,
        involved_tokens: vec![
            sol.to_string(),
            new_token.clone(),
        ],
        allowed_output_tokens: vec![
            sol.to_string(),
            new_token.clone(),
        ],
        allowed_programs: vec![
            raydium.to_string(),
            solana_sdk::system_program::id().to_string(),
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA".to_string(), // SPL Token
        ],
        max_instructions: 6,
    }
}

#[tokio::test]
async fn test_strategy_executor_initialization() {
    let (_kp, b58_key) = get_test_keypair();
    let rpc_url = get_test_rpc_url();
    
    // Create a transaction executor
    let tx_executor = TransactionExecutor::new(&rpc_url, &b58_key, true).unwrap();
    assert!(tx_executor.simulation_mode());
    
    // Create a strategy executor
    let strategy_executor = StrategyExecutor::new(tx_executor);
    
    // Verify that we can access the executor through the getter
    let executor = strategy_executor.get_executor();
    assert!(executor.simulation_mode());
}

#[tokio::test]
async fn test_arbitrage_transaction_building() {
    let (_kp, b58_key) = get_test_keypair();
    let rpc_url = get_test_rpc_url();
    
    // Create a transaction executor in simulation mode
    let tx_executor = TransactionExecutor::new(&rpc_url, &b58_key, true).unwrap();
    
    // Create a strategy executor
    let strategy_executor = StrategyExecutor::new(tx_executor);
    
    // Create an arbitrage opportunity
    let opportunity = create_arbitrage_opportunity();
    
    // Execute the opportunity (this will internally build a transaction)
    let result = strategy_executor.execute_opportunity(&opportunity).await;
    
    // Verify that the opportunity was executed successfully
    assert!(result.is_ok(), "Failed to execute arbitrage opportunity: {:?}", result.err());
}

#[tokio::test]
async fn test_sandwich_transaction_building() {
    let (_kp, b58_key) = get_test_keypair();
    let rpc_url = get_test_rpc_url();
    
    // Create a transaction executor in simulation mode
    let tx_executor = TransactionExecutor::new(&rpc_url, &b58_key, true).unwrap();
    
    // Create a strategy executor
    let strategy_executor = StrategyExecutor::new(tx_executor);
    
    // Create a sandwich opportunity
    let opportunity = create_sandwich_opportunity();
    
    // Execute the opportunity (this will internally build a transaction)
    let result = strategy_executor.execute_opportunity(&opportunity).await;
    
    // Verify that the opportunity was executed successfully
    assert!(result.is_ok(), "Failed to execute sandwich opportunity: {:?}", result.err());
}

#[tokio::test]
async fn test_token_snipe_transaction_building() {
    let (_kp, b58_key) = get_test_keypair();
    let rpc_url = get_test_rpc_url();
    
    // Create a transaction executor in simulation mode
    let tx_executor = TransactionExecutor::new(&rpc_url, &b58_key, true).unwrap();
    
    // Create a strategy executor
    let strategy_executor = StrategyExecutor::new(tx_executor);
    
    // Create a token snipe opportunity
    let opportunity = create_token_snipe_opportunity();
    
    // Execute the opportunity (this will internally build a transaction)
    let result = strategy_executor.execute_opportunity(&opportunity).await;
    
    // Verify that the opportunity was executed successfully
    assert!(result.is_ok(), "Failed to execute token snipe opportunity: {:?}", result.err());
}

#[tokio::test]
async fn test_execute_opportunity_in_simulation_mode() {
    let (_kp, b58_key) = get_test_keypair();
    let rpc_url = get_test_rpc_url();
    
    // Create a transaction executor in simulation mode
    let tx_executor = TransactionExecutor::new(&rpc_url, &b58_key, true).unwrap();
    
    // Create a strategy executor
    let strategy_executor = StrategyExecutor::new(tx_executor);
    
    // Test with each opportunity type
    let opportunities = vec![
        create_arbitrage_opportunity(),
        create_sandwich_opportunity(),
        create_token_snipe_opportunity(),
    ];
    
    for opportunity in opportunities {
        // Execute the opportunity
        let result = strategy_executor.execute_opportunity(&opportunity).await;
        
        // In simulation mode, execution should succeed
        assert!(result.is_ok(), "Failed to execute opportunity in simulation mode: {:?}", result.err());
        
        // Should have returned a simulated signature
        let signature = result.unwrap();
        assert_ne!(signature, solana_sdk::signature::Signature::default());
    }
}

#[tokio::test]
async fn test_strategy_execution_service_trait_implementation() {
    let (_kp, b58_key) = get_test_keypair();
    let rpc_url = get_test_rpc_url();
    
    // Create a transaction executor in simulation mode
    let tx_executor = TransactionExecutor::new(&rpc_url, &b58_key, true).unwrap();
    
    // Create a strategy executor
    let strategy_executor = Arc::new(StrategyExecutor::new(tx_executor));
    
    // Create an opportunity
    let opportunity = create_arbitrage_opportunity();
    
    // Use the trait method to execute the opportunity
    use crate::evaluator::StrategyExecutionService;
    let result = strategy_executor.execute_opportunity(&opportunity).await;
    
    // In simulation mode, execution should succeed
    assert!(result.is_ok(), "Failed to execute opportunity through trait interface: {:?}", result.err());
} 