mod config;
mod error;
mod evaluator;
mod listen_bot;
mod rig_agent;
mod monitoring;
mod executor;

use crate::config::Settings;
use crate::error::{Result, SandoError};
use crate::monitoring::init_logging;
use crate::evaluator::{MevStrategyEvaluator, OpportunityEvaluator, arbitrage::ArbitrageEvaluator, arbitrage::ArbitrageConfig};
use crate::executor::TransactionExecutor;
use crate::rig_agent::RigAgent;
use tracing::{info, error};
use dotenv::dotenv;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file first
    dotenv().ok();

    // Initialize logging
    let log_dir = "./logs";
    let file_level = "debug";
    let console_level = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    
    let _guard = init_logging(log_dir, &file_level, &console_level)
        .map_err(|e| SandoError::InternalError(format!("Failed to initialize logging: {}", e)))?;

    // Remove the async move block, execute directly in main
    // let main_span = tracing::span!(Level::INFO, "main_execution");
    // async move {
    // Enter span if needed:
    let main_span = tracing::info_span!("main_execution");
    let _main_span_guard = main_span.enter(); // Keep guard

    info!("SandoSeer logging initialized. Console: {}, File: {} (in {})", console_level, file_level, log_dir);
    info!("Starting SandoSeer MEV Oracle...");

    // Load configuration
    let settings = Settings::from_env()
        .map_err(|e| SandoError::Config(e))?;

    info!("Configuration loaded successfully");
    info!("Connected to Solana RPC: {}", settings.solana_rpc_url);
    info!("Risk level set to: {:?}", settings.risk_level);

    // Initialize components
    info!("Initializing components...");
    let rig_agent = RigAgent::from_env()?;
    let executor = TransactionExecutor::new(
        &settings.solana_rpc_url,
        &settings.wallet_private_key,
        true // Force simulation mode for now
    )?;
    
    // Initialize the main evaluator
    let mut evaluator = OpportunityEvaluator::new_with_thresholds(
        settings.min_profit_threshold, 
        settings.risk_level.into(),
        settings.min_profit_threshold,
        Default::default() 
    ).await?;

    // Initialize and register strategy-specific evaluators
    let arbitrage_config = ArbitrageConfig::default(); // Use default config for now
    let arbitrage_evaluator = ArbitrageEvaluator::new(arbitrage_config);
    evaluator.register_evaluator(Box::new(arbitrage_evaluator));
    
    // TODO: Initialize and register other evaluators (Sandwich, Snipe) when implemented

    info!("Components initialized.");

    // Run Test Pipeline
    info!("Running test pipeline...");
    if let Err(e) = run_test_pipeline(rig_agent, evaluator, executor).await {
        error!(error = %e, "Test pipeline failed");
    } else {
        info!("Test pipeline completed.");
    }

    // TODO: Replace test pipeline with actual main loop (Listen Bot -> ...)

    info!("Exiting test run."); 

    // }.instrument(main_span).await;
    // Span guard _main_span_guard drops here

    info!("SandoSeer shutting down...");
    // The logging _guard will be dropped here, flushing file logs
    Ok(())
}

/// Temporary function to test the component flow
async fn run_test_pipeline(
    rig_agent: RigAgent,
    evaluator: OpportunityEvaluator,
    executor: TransactionExecutor
) -> Result<()> {
    info!(target: "test_pipeline", "Creating mock data...");
    // Mock data for RIG Agent (kept for potential future use/context)
    let mock_opportunity_for_rig = crate::rig_agent::RawOpportunityData {
        source_dex: "MockDex".to_string(),
        transaction_hash: "mock_tx_hash_abc".to_string(),
        input_token: "SOL".to_string(),
        output_token: "USDC".to_string(),
        input_amount: 10.0,
        output_amount: 1600.0, // 10 SOL -> 1600 USDC
    };
    let mock_context_for_rig = crate::rig_agent::MarketContext {
        input_token_price_usd: 160.0,
        output_token_price_usd: 1.0,
        pool_liquidity_usd: 1000000.0,
        recent_volatility_percent: 0.5,
    };

    // Mock data specifically for ArbitrageEvaluator::evaluate
    let mock_arbitrage_data = json!({
        "token_path": ["USDC", "SOL", "USDC"], // Example path
        "amounts": [1000.0, 6.25, 1005.0], // Input USDC, intermediate SOL, Output USDC
        "price_impacts": [0.001, 0.001], // Impact for each step
        "liquidity": [500000.0, 500000.0], // Liquidity for each pool
        "dexes": ["Orca", "Raydium"] // DEXes for each step
    });

    // 1. Call RIG Agent (optional for this flow, but kept)
    info!(target: "test_pipeline", "Calling RIG Agent (for context)...");
    let ai_evaluation = rig_agent.evaluate_opportunity(&mock_opportunity_for_rig, &mock_context_for_rig).await?;
    info!(target: "test_pipeline", ai_evaluation = ?ai_evaluation, "Received AI evaluation (context)");

    // 2. Evaluate with OpportunityEvaluator using strategy-specific data
    info!(target: "test_pipeline", "Calling OpportunityEvaluator with mock arbitrage data...");
    let opportunities = evaluator.evaluate_opportunity(mock_arbitrage_data).await?;

    if opportunities.is_empty() {
        info!(target: "test_pipeline", "No viable opportunities found by the evaluator.");
        return Ok(());
    }

    // 3. Process the first evaluated opportunity
    // (In a real scenario, might loop or prioritize)
    let first_opportunity = &opportunities[0];
    info!(target: "test_pipeline", opportunity = ?first_opportunity, "Evaluated opportunity details");

    // Use the decision made by the OpportunityEvaluator
    let final_decision = first_opportunity.decision.unwrap_or(crate::evaluator::ExecutionDecision::Decline);
    info!(target: "test_pipeline", final_decision = ?final_decision, "Evaluator decision made");

    // 4. Execute if viable
    if final_decision == crate::evaluator::ExecutionDecision::Execute {
        info!(target: "test_pipeline", "Decision is EXECUTE. Proceeding to executor...");
        
        // TODO: Extract actual path/strategy from opportunity.metadata
        // For now, continue using the dummy path for the executor test
        let mut mock_arbitrage_path = crate::executor::ArbitragePath {
            steps: vec![]
        };
        if mock_arbitrage_path.steps.is_empty() {
            let dummy_pubkey = solana_sdk::pubkey::Pubkey::new_unique();
             mock_arbitrage_path.steps.push(crate::executor::ArbitrageStep {
                dex_program_id: dummy_pubkey,
                input_token_mint: dummy_pubkey, 
                output_token_mint: dummy_pubkey,
                input_amount: 1000_000_000, // Example amount (e.g., 1000 USDC)
                min_output_amount: 0 // Slippage control needed here
            });
        }
        
        let transaction_to_execute = executor.build_arbitrage_transaction(&mock_arbitrage_path)?;
        
        // Execute (in simulation mode)
        let execution_result = executor.execute_transaction(transaction_to_execute).await;
        match execution_result {
            Ok(signature) => info!(target: "test_pipeline", signature = %signature, "Transaction executed/simulated successfully."),
            Err(e) => error!(target: "test_pipeline", error = %e, "Transaction execution/simulation failed."),
        }
    } else {
        info!(target: "test_pipeline", "Decision was not EXECUTE. No execution attempted.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_main_setup() {
        std::env::set_var("SOLANA_RPC_URL", "https://test.solana.com");
        std::env::set_var("WALLET_PRIVATE_KEY", "test_key");
        std::env::set_var("RIG_API_KEY", "test_rig_key");
        std::env::set_var("MAX_CONCURRENT_TRADES", "5");
        std::env::set_var("MIN_PROFIT_THRESHOLD", "0.01");
        std::env::set_var("MAX_SLIPPAGE", "0.02");
        std::env::set_var("LOG_LEVEL", "info");
        std::env::set_var("ORCA_API_URL", "https://test.orca.so");
        std::env::set_var("RAYDIUM_API_URL", "https://test.raydium.io");
        std::env::set_var("JUPITER_API_URL", "https://test.jup.ag");
        std::env::set_var("MAX_POSITION_SIZE", "1000");
        std::env::set_var("RISK_LEVEL", "medium");

        let settings = Settings::from_env().unwrap();
        assert_eq!(settings.solana_rpc_url, "https://test.solana.com");
    }
}
