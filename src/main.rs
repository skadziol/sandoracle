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
use crate::evaluator::OpportunityEvaluator;
use crate::executor::TransactionExecutor;
use crate::rig_agent::RigAgent;
use tracing::{info, error};
use dotenv::dotenv;

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
    let evaluator = OpportunityEvaluator::new_with_thresholds(
        settings.min_profit_threshold, // Using general threshold for now
        settings.risk_level.into(), // Will use .into() after implementing From
        settings.min_profit_threshold,
        Default::default() // Using default detailed thresholds for now
    ).await?;
    // TODO: Initialize Listen Bot
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
    // Mock data (adjust as needed)
    let mock_opportunity = crate::rig_agent::RawOpportunityData {
        source_dex: "MockDex".to_string(),
        transaction_hash: "mock_tx_hash_abc".to_string(),
        input_token: "SOL".to_string(),
        output_token: "USDC".to_string(),
        input_amount: 10.0,
        output_amount: 1600.0, // 10 SOL -> 1600 USDC
    };
    let mock_context = crate::rig_agent::MarketContext {
        input_token_price_usd: 160.0,
        output_token_price_usd: 1.0,
        pool_liquidity_usd: 1000000.0,
        recent_volatility_percent: 0.5,
    };

    // 1. Evaluate with RIG Agent
    info!(target: "test_pipeline", "Calling RIG Agent...");
    let ai_evaluation = rig_agent.evaluate_opportunity(&mock_opportunity, &mock_context).await?; 
    info!(target: "test_pipeline", ai_evaluation = ?ai_evaluation, "Received AI evaluation");

    // 2. Evaluate with OpportunityEvaluator (using AI output)
    // TODO: Refactor OpportunityEvaluator to properly take AI input
    // For now, we simulate its decision based on AI output + thresholds
    info!(target: "test_pipeline", "Calling Opportunity Evaluator (simulated decision)...");
    let final_decision = if ai_evaluation.is_viable 
                           && ai_evaluation.confidence_score >= evaluator.min_confidence()
                           // && ai_evaluation.estimated_profit_usd >= evaluator.min_profit_threshold // Maybe check this?
                        {
                            // Placeholder: Assume AI suggestion is trustworthy if confidence is high
                            match ai_evaluation.suggested_action.as_str() {
                                "Execute Arbitrage" => crate::evaluator::ExecutionDecision::Execute,
                                "Execute Sandwich" => crate::evaluator::ExecutionDecision::Execute,
                                _ => crate::evaluator::ExecutionDecision::Decline, 
                            }
                        } else {
                             crate::evaluator::ExecutionDecision::Decline
                        };
    info!(target: "test_pipeline", final_decision = ?final_decision, "Evaluator decision made");

    // 3. Execute if viable
    if final_decision == crate::evaluator::ExecutionDecision::Execute {
        info!(target: "test_pipeline", "Decision is EXECUTE. Proceeding to executor...");
        // Build mock transaction (based on mock opportunity)
        // TODO: Extract actual path/strategy from AI evaluation
        let mut mock_arbitrage_path = crate::executor::ArbitragePath {
            steps: vec![/* TODO: Populate with mock steps based on mock_opportunity */]
        };
        // Add a dummy step if empty to avoid error in build_arbitrage_transaction
        if mock_arbitrage_path.steps.is_empty() {
             // Error: Cannot create Pubkey from str "dummy"
             // let dummy_pubkey = solana_sdk::pubkey::Pubkey::from_str("dummy"); 
             // Using new_unique instead
            let dummy_pubkey = solana_sdk::pubkey::Pubkey::new_unique();
             mock_arbitrage_path.steps.push(crate::executor::ArbitrageStep {
                dex_program_id: dummy_pubkey,
                input_token_mint: dummy_pubkey, 
                output_token_mint: dummy_pubkey,
                input_amount: 0,
                min_output_amount: 0
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
        info!(target: "test_pipeline", "Decision is DECLINE. No execution attempted.");
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
