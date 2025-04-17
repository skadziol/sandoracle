// Integration tests for MEV strategy execution

use sandoracle::{
    executor::{TransactionExecutor, strategies::StrategyExecutor},
    evaluator::{MevOpportunity, MevStrategy, RiskLevel, StrategyExecutionService, OpportunityEvaluator},
    config::Settings,
};
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::{Keypair, Signer};
use std::sync::Arc;
use std::str::FromStr;
use serde_json::json;

// Helper to build a full test environment
async fn setup_test_environment() -> (Arc<StrategyExecutor>, Arc<OpportunityEvaluator>) {
    // Create a keypair for testing
    let keypair = Keypair::new();
    let b58_key = bs58::encode(keypair.to_bytes()).into_string();
    
    // Use devnet for testing
    let rpc_url = "https://api.devnet.solana.com".to_string();
    
    // Create a transaction executor in simulation mode
    let tx_executor = TransactionExecutor::new(&rpc_url, &b58_key, true).unwrap();
    
    // Create a strategy executor
    let strategy_executor = Arc::new(StrategyExecutor::new(tx_executor));
    
    // Create an opportunity evaluator
    let evaluator = OpportunityEvaluator::new_with_thresholds(
        50.0,                // min_profit_threshold
        RiskLevel::Medium,   // max_risk_level
        0.05,                // min_profit_percentage
        Default::default(),  // Default execution thresholds
    )
    .await
    .unwrap();
    
    // Set the strategy executor on the evaluator
    let mut evaluator = evaluator;
    evaluator.set_strategy_executor(strategy_executor.clone()).await;
    
    (strategy_executor, Arc::new(evaluator))
}

// Helper to create a test arbitrage opportunity
fn create_test_arbitrage_opportunity() -> MevOpportunity {
    // Example token addresses (using mainnet tokens for realism)
    let sol = "So11111111111111111111111111111111111111112"; // SOL
    let usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"; // USDC
    let usdt = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"; // USDT
    
    // DEX program IDs (using real program IDs for realism)
    let raydium = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"; // Raydium
    let orca = "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP"; // Orca
    
    // Create arbitrage metadata
    let metadata = json!({
        "token_path": [sol, usdc, usdt, sol],
        "prices": [100.0, 1.0, 1.0, 101.0],
        "liquidity": [100000.0, 200000.0, 200000.0, 100000.0],
        "price_impacts": [0.001, 0.001, 0.001, 0.001],
        "dexes": [raydium, orca, raydium]
    });
    
    MevOpportunity {
        strategy: MevStrategy::Arbitrage,
        estimated_profit: 1.0, // 1 SOL profit
        confidence: 0.95,
        risk_level: RiskLevel::Low,
        required_capital: 100.0, // 100 SOL
        execution_time: 500, // 500ms
        metadata,
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

#[tokio::test]
async fn test_strategy_evaluator_integration() {
    // Setup the test environment
    let (strategy_executor, evaluator) = setup_test_environment().await;
    
    // Create a test arbitrage opportunity
    let mut opportunity = create_test_arbitrage_opportunity();
    
    // Process the opportunity through the evaluator
    let result = evaluator.process_mev_opportunity(&mut opportunity).await;
    
    // In simulation mode, this should succeed but not actually execute
    assert!(result.is_ok());
    
    // The opportunity should now have a decision
    assert!(opportunity.decision.is_some());
}

#[tokio::test]
async fn test_execution_strategy_service_trait() {
    // Setup the test environment
    let (strategy_executor, _) = setup_test_environment().await;
    
    // Create a test arbitrage opportunity
    let opportunity = create_test_arbitrage_opportunity();
    
    // Execute the opportunity via the trait interface
    let result = StrategyExecutionService::execute_opportunity(
        strategy_executor.as_ref(), 
        &opportunity
    ).await;
    
    // In simulation mode, this should succeed
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_full_execution_pipeline() {
    // Setup the test environment
    let (strategy_executor, evaluator) = setup_test_environment().await;
    
    // Create a test arbitrage opportunity
    let mut opportunity = create_test_arbitrage_opportunity();
    
    // Step 1: Calculate opportunity score
    let score = evaluator.calculate_opportunity_score(&opportunity).await;
    opportunity.score = Some(score);
    
    // Step 2: Make execution decision
    let decision = evaluator.make_execution_decision(&opportunity).await;
    opportunity.decision = Some(decision);
    
    // Step 3: Check if we should execute
    let should_execute = evaluator.should_execute(&opportunity).await;
    
    // Step 4: Execute if appropriate
    let result = if should_execute {
        Some(strategy_executor.execute_opportunity(&opportunity).await)
    } else {
        None
    };
    
    // In test mode with a good opportunity, we should have tried to execute
    assert!(should_execute);
    assert!(result.is_some());
    assert!(result.unwrap().is_ok());
} 