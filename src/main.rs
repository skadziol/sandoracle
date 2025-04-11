mod config;
mod error;
mod evaluator;
mod listen_bot;
mod rig_agent;
mod monitoring;
mod executor;
mod market_data;
mod jupiter_client;

use crate::config::Settings;
use crate::error::{Result as SandoResult, SandoError};
use crate::monitoring::init_logging;
use crate::monitoring::log_utils::{check_log_directory, rotate_logs, clear_debug_logs};
use crate::listen_bot::{ListenBot, ListenBotCommand};
use crate::evaluator::{OpportunityEvaluator, ExecutionThresholds};
use crate::executor::{TransactionExecutor, ExecutionService};
use tracing::{info, error};
use dotenv::dotenv;
use tokio::signal;
use std::sync::Arc;
use crate::config::RiskLevel as ConfigRiskLevel;
use crate::evaluator::RiskLevel as EvalRiskLevel;

#[tokio::main]
async fn main() -> SandoResult<()> {
    // Load .env file first
    dotenv().ok();

    // Initialize logging
    let log_dir = "./logs";
    let file_level = std::env::var("FILE_LOG_LEVEL").unwrap_or_else(|_| "info".to_string());
    let console_level = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    
    // Rotate logs before starting a new session
    if let Err(e) = rotate_logs(log_dir) {
        eprintln!("Warning: Failed to rotate log files: {}", e);
    }
    
    let _guard = init_logging(log_dir, &file_level, &console_level)?;

    // Check log directory size
    if let Err(e) = check_log_directory(log_dir) {
        error!(error = %e, "Failed to check log directory");
    }
    
    // Clean up old debug logs if running at info level or higher
    if file_level != "debug" && file_level != "trace" {
        if let Err(e) = clear_debug_logs(log_dir) {
            error!(error = %e, "Failed to clear debug logs");
        }
    }
    
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
    
    // Initialize Transaction Executor
    info!("Initializing TransactionExecutor...");
    let transaction_executor = TransactionExecutor::new(
        &settings.solana_rpc_url,
        &settings.wallet_private_key,
        settings.simulation_mode,
    )
    .map_err(|e| SandoError::DependencyError(format!("Failed to create TransactionExecutor: {}", e)))?;
    
    // Initialize Strategy Execution Service
    info!("Initializing ExecutionService...");
    let execution_service = ExecutionService::new(transaction_executor.clone())
        .with_max_retries(settings.max_retries.unwrap_or(3))
        .with_simulation(!settings.simulation_mode); // Only simulate if not in simulation mode
    
    // Wrap the execution service in an Arc for safe sharing between threads
    let execution_service_arc = Arc::new(execution_service);
    
    // Initialize the OpportunityEvaluator
    info!("Initializing OpportunityEvaluator...");
    
    // Convert the risk level from config to evaluator type
    let eval_risk_level = evaluator::RiskLevel::from(settings.risk_level.clone());
    
    let mut evaluator = OpportunityEvaluator::new_with_thresholds(
        settings.min_profit_threshold,  // Use settings for minimum profit threshold
        eval_risk_level,                // Use converted risk level
        settings.min_profit_threshold,  // Duplicate setting as a fallback
        ExecutionThresholds::default(), // Use default thresholds for now
    )
    .await
    .map_err(|e| SandoError::DependencyError(format!("Failed to create OpportunityEvaluator: {}", e)))?;
    
    // Set the execution service on the evaluator
    evaluator.set_strategy_executor(execution_service_arc.clone()).await;

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
        wallet_pubkey = %transaction_executor.signer_pubkey(),
        simulation_mode = settings.simulation_mode,
        "All components initialized successfully"
    );

    // Start the ListenBot in its own task
    let listen_bot_handle = tokio::spawn(async move {
        if let Err(e) = listen_bot.start().await {
            error!(error = %e, "ListenBot failed to start or encountered an error");
        }
    });

    // Spawn a task to periodically check and clean logs
    let log_dir_clone = log_dir.to_string();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(3600)); // Every hour
        loop {
            interval.tick().await;
            info!(target: "log_management", "Running scheduled log cleanup...");
            
            if let Err(e) = rotate_logs(&log_dir_clone) {
                error!(target: "log_management", error = %e, "Failed to rotate log files");
            }
            
            if let Err(e) = check_log_directory(&log_dir_clone) {
                error!(target: "log_management", error = %e, "Failed to check log directory");
            }
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
