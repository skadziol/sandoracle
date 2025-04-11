use sandoseer::evaluator::{MevOpportunity, MevStrategy, RiskLevel, StrategyExecutionService};
use sandoseer::executor::{ExecutionService, TransactionExecutor};
use sandoseer::config::Settings;
use serde_json::{json, Value};
use std::sync::Arc;
use std::str::FromStr;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::commitment_config::CommitmentConfig;

// Helper function to create a test opportunity
fn create_test_opportunity(strategy: MevStrategy) -> MevOpportunity {
    let metadata = match strategy {
        MevStrategy::Arbitrage => json!({
            "token_path": ["USDC", "SOL", "USDC"],
            "price_difference_percent": 2.5,
            "price_impact_percent": 0.5,
            "source_dex": "Jupiter",
            "target_dex": "Raydium",
            "estimated_gas_cost_usd": 0.02,
            "optimal_trade_size_usd": 1000.0
        }),
        MevStrategy::Sandwich => json!({
            "target_tx_hash": "3m9PXbYfAiJexGVFNAkDXtw3uJN9KCT2JNbhwf7CuJGS6wnG7Wk3W1BKfXKbshfyzCjmfHMcY8Q",
            "token_pair": ["USDC", "SOL"],
            "dex": "Jupiter",
            "target_tx_value_usd": 5000.0,
            "pool_liquidity_usd": 1000000.0,
            "price_impact_pct": 0.3,
            "optimal_position_size_usd": 200.0,
            "frontrun_slippage_pct": 0.2,
            "backrun_slippage_pct": 0.3
        }),
        MevStrategy::TokenSnipe => json!({
            "token_address": format!("{}NEWTOKEN", Pubkey::new_unique()),
            "token_symbol": "NEWTKN",
            "dex": "Jupiter",
            "initial_liquidity_usd": 50000.0,
            "initial_price_usd": 0.0001,
            "initial_market_cap_usd": 200000.0,
            "proposed_investment_usd": 500.0,
            "expected_return_multiplier": 3.0,
            "max_position_percent": 0.5,
            "optimal_hold_time_seconds": 1800,
            "acquisition_supply_percent": 0.25
        })
    };

    MevOpportunity {
        strategy,
        estimated_profit: 100.0,
        confidence: 0.85,
        risk_level: RiskLevel::Medium,
        required_capital: 1000.0,
        execution_time: 200,
        metadata,
        score: Some(0.75),
        decision: None,
        involved_tokens: vec!["SOL".to_string(), "USDC".to_string()],
        allowed_output_tokens: vec!["USDC".to_string()],
        allowed_programs: vec![],
        max_instructions: 200,
    }
}

// Helper function to create test settings
fn create_test_settings() -> Settings {
    Settings {
        solana_rpc_url: "https://api.testnet.solana.com".to_string(),
        solana_ws_url: "wss://api.testnet.solana.com".to_string(),
        commitment: CommitmentConfig::confirmed(),
        wallet_private_key: "Test1111111111111111111111111111111111111111111".to_string(),
        simulation_mode: true,
        rig_api_key: Some("test_api_key".to_string()),
        rig_model_provider: Some("anthropic".to_string()),
        rig_model_name: Some("claude-3-opus".to_string()),
        telegram_bot_token: Some("test_bot_token".to_string()),
        telegram_chat_id: Some("test_chat_id".to_string()),
        log_level: "info".to_string(),
        risk_level: sandoseer::config::RiskLevel::Medium,
        max_concurrent_trades: 3,
        min_profit_threshold: 10.0,
        max_slippage: 0.01,
        max_position_size: 1000.0,
        allowed_tokens: Some(vec!["SOL".to_string(), "USDC".to_string(), "ETH".to_string()]),
        blocked_tokens: Some(vec![]),
        request_timeout_secs: Some(30),
        max_retries: Some(3),
        retry_backoff_ms: Some(500),
        orca_api_url: Some("https://api.testnet.orca.so".to_string()),
        raydium_api_url: Some("https://api.testnet.raydium.io".to_string()),
        jupiter_api_url: Some("https://quotes.jup.ag/v4".to_string()),
        arbitrage_min_profit: Some(10.0),
        sandwich_min_profit: Some(20.0),
        snipe_min_profit: Some(50.0),
        geyser_plugin_endpoint_url: "ws://localhost:8900/".to_string(),
        whitelisted_dexes: Some(vec!["Jupiter".to_string(), "Orca".to_string(), "Raydium".to_string()]),
    }
}

#[tokio::test]
async fn test_execution_service_validation() {
    // This test verifies that the validation logic in ExecutionService works correctly
    // We'll need to mock the TransactionExecutor for testing, but in this case we can
    // use the simulation_mode feature to avoid actually executing transactions

    // Create our own test settings
    let _settings = create_test_settings();
    
    // Create a mock transaction executor
    let tx_executor = mock_transaction_executor();
    
    // Create the ExecutionService
    let execution_service = ExecutionService::new(tx_executor)
        .with_simulation(true);
    
    // Test with an arbitrage opportunity
    let arb_opportunity = create_test_opportunity(MevStrategy::Arbitrage);
    
    // Instead of calling private method directly, we'll use the public execute_opportunity
    // method which will internally validate the opportunity
    let result = execution_service.execute_opportunity(&arb_opportunity).await;
    assert!(result.is_err(), "Expected error for mocked execution");
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("mock") || err_msg.contains("test"), 
            "Error should be from mock: {}", err_msg);
    
    // Test with a low profit opportunity
    let mut low_profit_opportunity = create_test_opportunity(MevStrategy::Arbitrage);
    low_profit_opportunity.estimated_profit = -5.0;
    let result = execution_service.execute_opportunity(&low_profit_opportunity).await;
    assert!(result.is_err(), "Low profit opportunity should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("profit") || err_msg.contains("validation"), 
            "Error should mention profit issue: {}", err_msg);
    
    // Test with high risk opportunity
    let mut high_risk_opportunity = create_test_opportunity(MevStrategy::TokenSnipe);
    high_risk_opportunity.risk_level = RiskLevel::High;
    let result = execution_service.execute_opportunity(&high_risk_opportunity).await;
    // High risk is a warning but not an error, so we still expect an error from the mock
    assert!(result.is_err(), "Expected error for mocked execution");
}

// Helper to create a mocked transaction executor
fn mock_transaction_executor() -> TransactionExecutor {
    // Since we can't easily mock the TransactionExecutor directly, we'll create a real one
    // in simulation mode - it will never execute real transactions
    TransactionExecutor::new(
        "https://api.testnet.solana.com",
        // Use a well-formed but invalid key so we don't trigger validation issues
        "4NMwxzmYbBq8YCXzBBKjHZYxWG3V3NAeREJXNunxJQov",
        true // simulation mode
    ).unwrap_or_else(|_| panic!("Failed to create test executor"))
}

// In a real implementation, we would have more tests to verify that:
// 1. The execution_with_retry logic works correctly with retries
// 2. The execute_opportunity method correctly delegates to StrategyExecutor
// 3. Integration with the OpportunityEvaluator

// Note: Implementing these tests would require more sophisticated mocking
// of the TransactionExecutor and StrategyExecutor classes 