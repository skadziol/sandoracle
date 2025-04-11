mod config;
mod error;
mod evaluator;
mod listen_bot;
mod rig_agent;
mod monitoring;
mod executor;

use crate::config::Settings;
use crate::error::{Result as SandoResult, SandoError};
use crate::monitoring::init_logging;
use crate::listen_bot::{ListenBot, ListenBotCommand};
use crate::evaluator::{OpportunityEvaluator, ExecutionThresholds, RiskLevel};
use tracing::{info, error};
use dotenv::dotenv;
use tokio::signal;
use std::sync::Arc;

#[tokio::main]
async fn main() -> SandoResult<()> {
    // Load .env file first
    dotenv().ok();

    // Initialize logging
    let log_dir = "./logs";
    let file_level = "debug";
    let console_level = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let _guard = init_logging(log_dir, file_level, &console_level)?;

    // Enter span if needed:
    let main_span = tracing::info_span!("main_execution");
    let _main_span_guard = main_span.enter(); // Keep guard

    info!("Starting SandoSeer MEV Oracle...");

    // Load configuration
    let settings = Settings::from_env()
        .map_err(|e| SandoError::Config(e))?;

    info!("Configuration loaded successfully");
    info!("Connected to Solana RPC: {}", settings.solana_rpc_url);
    info!("Risk level set to: {:?}", settings.risk_level);

    // Initialize components
    info!("Initializing components...");
    
    // Initialize the OpportunityEvaluator
    info!("Initializing OpportunityEvaluator...");
    let evaluator = OpportunityEvaluator::new_with_thresholds(
        settings.min_profit_threshold,  // Use settings for minimum profit threshold
        settings.risk_level,            // Use settings for risk level
        settings.min_profit_threshold,  // Duplicate setting as a fallback
        ExecutionThresholds::default(), // Use default thresholds for now
    )
    .await
    .map_err(|e| SandoError::DependencyError(format!("Failed to create OpportunityEvaluator: {}", e)))?;

    // Wrap the evaluator in an Arc for safe sharing between threads
    let evaluator_arc = Arc::new(evaluator);
    
    // Initialize ListenBot (this will internally init listen-engine)
    let (mut listen_bot, listen_bot_cmd_tx) = ListenBot::from_settings(&settings).await?;
    
    // Set the evaluator on the ListenBot
    listen_bot.set_evaluator(evaluator_arc.clone());

    // Log detailed information about the initialized components
    info!(
        min_confidence = evaluator_arc.min_confidence(),
        risk_level = ?settings.risk_level,
        min_profit = settings.min_profit_threshold,
        "OpportunityEvaluator initialized and connected to ListenBot"
    );

    info!("Components initialized successfully");

    // Start the ListenBot in its own task
    let listen_bot_handle = tokio::spawn(async move {
        if let Err(e) = listen_bot.start().await {
            error!(error = %e, "ListenBot failed to start or encountered an error");
        }
    });

    // Handle graceful shutdown (Ctrl+C)
    let shutdown_cmd_tx = listen_bot_cmd_tx.clone();
    tokio::spawn(async move {
        signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
        info!("Ctrl+C received. Sending shutdown signal to ListenBot...");
        if let Err(_) = shutdown_cmd_tx.send(ListenBotCommand::Shutdown).await {
            error!("Failed to send shutdown command to ListenBot.");
        }
    });

    info!("SandoSeer main loop running. Press Ctrl+C to exit.");

    // Wait for the listen_bot task to complete (it will run until shutdown)
    if let Err(e) = listen_bot_handle.await {
        error!(error = ?e, "ListenBot task failed or panicked");
    }

    // }.instrument(main_span).await;
    // Span guard _main_span_guard drops here

    info!("SandoSeer shutting down...");
    // The logging _guard will be dropped here, flushing file logs
    Ok(())
}

// Removed Test Pipeline Function
// /// Temporary function to test the component flow
// async fn run_test_pipeline(
//     rig_agent: RigAgent,
//     evaluator: OpportunityEvaluator,
//     executor: TransactionExecutor
// ) -> Result<()> { ... }

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
