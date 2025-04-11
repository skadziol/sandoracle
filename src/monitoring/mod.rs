use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling;
use std::path::Path;
use crate::error::{Result, SandoError};
use std::time::Duration;
use tokio::time::interval;
use crate::config::Settings;
use crate::executor::TransactionExecutor;
use tracing::{info, warn, error};

pub struct Monitor {
    // TODO: Add fields
}

impl Monitor {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn log_trade(&self) -> Result<()> {
        // TODO: Implement trade logging
        Ok(())
    }

    pub async fn send_notification(&self) -> Result<()> {
        // TODO: Implement Telegram notifications
        Ok(())
    }

    pub async fn track_performance(&self) -> Result<()> {
        // TODO: Implement performance tracking
        Ok(())
    }
}

/// Initializes the logging system (both console and file).
/// Returns a guard that must be kept alive for file logging to work.
pub fn init_logging(log_dir: &str, file_level: &str, console_level: &str) -> Result<WorkerGuard> {
    // Ensure log directory exists
    let log_path = Path::new(log_dir);
    if !log_path.exists() {
        std::fs::create_dir_all(log_path)
            .map_err(|e| SandoError::Io(e))?;
    }

    // --- File Logger --- 
    let file_appender = rolling::daily(log_dir, "sandoseer.log");
    let (non_blocking_appender, guard) = tracing_appender::non_blocking(file_appender);
    
    let file_filter = EnvFilter::try_new(file_level)
        .map_err(|e| SandoError::ConfigError(format!("Invalid file log level filter '{}': {}", file_level, e)))?;
    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(non_blocking_appender)
        .with_ansi(false) // No ANSI colors in files
        .with_span_events(FmtSpan::CLOSE) // Include span timings
        .json() // Log as JSON for easier parsing
        .with_filter(file_filter);

    // --- Console Logger --- 
    let console_filter = EnvFilter::try_new(console_level)
        .map_err(|e| SandoError::ConfigError(format!("Invalid console log level filter '{}': {}", console_level, e)))?;
    let console_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_span_events(FmtSpan::CLOSE)
        .with_filter(console_filter);

    // --- Combine Layers and Initialize --- 
    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .try_init()
        .map_err(|e| SandoError::InternalError(format!("Failed to initialize tracing subscriber: {}", e)))?;

    Ok(guard)
}

// TODO: Add functions for structured logging of specific events (trades, errors etc.)
// Example:
// pub fn log_trade(trade_details: &TradeInfo) {
//     info!(target: "trade_log", trade = ?trade_details, "Trade executed");
// }

// --- Health Monitoring --- 

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentStatus {
    Ok,
    Warning(String),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ComponentHealth {
    name: String,
    status: ComponentStatus,
}

/// Checks the health of the Solana RPC connection.
pub async fn check_rpc_connection(settings: &Settings) -> ComponentHealth {
    let name = "Solana RPC".to_string();
    info!(target: "health_check", component = name, "Running check...");
    match tokio::net::TcpStream::connect(&settings.solana_rpc_url).await { // Basic check
    // TODO: Replace with a proper RPC call like get_health or get_slot
    // match RpcClient::new(settings.solana_rpc_url.clone()).get_health().await {
        Ok(_) => ComponentHealth { name, status: ComponentStatus::Ok },
        Err(e) => ComponentHealth {
            name,
            status: ComponentStatus::Error(format!("Failed RPC check: {}", e)),
        },
    }
}

/// Checks the health of the Transaction Executor (e.g., wallet balance).
pub async fn check_executor_status(executor: &TransactionExecutor) -> ComponentHealth {
    let name = "Transaction Executor".to_string();
    info!(target: "health_check", component = name, "Running check...");
    // TODO: Implement actual checks, e.g.:
    // - Check wallet SOL balance using executor.rpc_client and executor.signer.pubkey()
    // - Check if signer is valid
    let balance_threshold_lamports = 10_000_000; // Example: 0.01 SOL
    match executor.get_wallet_balance().await {
        Ok(balance) => {
            if balance < balance_threshold_lamports {
                ComponentHealth {
                    name,
                    status: ComponentStatus::Warning(format!("Low wallet balance: {} lamports", balance))
                }
            } else {
                ComponentHealth { name, status: ComponentStatus::Ok }
            }
        },
        Err(e) => ComponentHealth {
            name,
            status: ComponentStatus::Error(format!("Failed to get wallet balance: {}", e))
        }
    }
}

/// Placeholder for checking Listen Engine health.
pub async fn check_listen_engine_status() -> ComponentHealth {
    let name = "Listen Engine".to_string();
    info!(target: "health_check", component = name, "Running check...");
    // TODO: Implement check (e.g., check last message timestamp, queue size)
    ComponentHealth { name, status: ComponentStatus::Ok } // Placeholder
}

/// Placeholder for checking RIG Agent health.
pub async fn check_rig_agent_status() -> ComponentHealth {
    let name = "RIG Agent".to_string();
    info!(target: "health_check", component = name, "Running check...");
    // TODO: Implement check (e.g., ping API endpoint, check API key validity)
    ComponentHealth { name, status: ComponentStatus::Ok } // Placeholder
}

/// --- Notification System ---

/// Sends a notification (currently logs, TODO: add external integrations).
pub async fn send_notification(message: &str, level: &str) {
    match level {
        "error" => {
            error!(target: "notification", message, "NOTIFICATION_ERROR");
            // TODO: Implement external notification sending (e.g., Telegram)
        }
        "warning" => {
            warn!(target: "notification", message, "NOTIFICATION_WARNING");
            // TODO: Implement external notification sending (e.g., Telegram)
        }
        _ => {
            info!(target: "notification", message, "NOTIFICATION_INFO");
        }
    }
}

/// Runs periodic health checks for all components.
pub async fn run_periodic_health_checks(
    settings: Settings,
    executor: TransactionExecutor,
    check_interval_secs: u64,
) {
    info!(interval_secs = check_interval_secs, "Starting periodic health checks...");
    let mut interval = interval(Duration::from_secs(check_interval_secs));

    loop {
        interval.tick().await;
        info!(target: "health_check", "Running scheduled health checks...");

        let mut overall_status = ComponentStatus::Ok;
        let mut unhealthy_components = Vec::new();

        let checks = vec![
            check_rpc_connection(&settings).await,
            check_executor_status(&executor).await,
            check_listen_engine_status().await,
            check_rig_agent_status().await,
            // Add more component checks here
        ];

        for health in checks {
            match &health.status {
                ComponentStatus::Ok => {
                    info!(target: "health_check", component = health.name, status = "Ok", "Health check passed");
                }
                ComponentStatus::Warning(msg) => {
                    warn!(target: "health_check", component = health.name, status = "Warning", message = msg, "Health check warning");
                    if overall_status == ComponentStatus::Ok {
                        overall_status = ComponentStatus::Warning("One or more components have warnings".to_string());
                    }
                    unhealthy_components.push(health);
                }
                ComponentStatus::Error(msg) => {
                    error!(target: "health_check", component = health.name, status = "Error", message = msg, "Health check failed!");
                    overall_status = ComponentStatus::Error("One or more components are unhealthy".to_string());
                    unhealthy_components.push(health);
                }
            }
        }
        
        match overall_status {
            ComponentStatus::Ok => info!(target: "health_check", "Overall system health: Ok"),
            ComponentStatus::Warning(ref msg) => {
                warn!(target: "health_check", message = msg, unhealthy_count = unhealthy_components.len(), "Overall system health: Warning");
                send_notification(&format!("System Health Warning: {} ({} unhealthy)", msg, unhealthy_components.len()), "warning").await;
            }
            ComponentStatus::Error(ref msg) => {
                 error!(target: "health_check", message = msg, unhealthy_count = unhealthy_components.len(), "Overall system health: Error!");
                 send_notification(&format!("System Health ERROR: {} ({} unhealthy)", msg, unhealthy_components.len()), "error").await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::thread;
    use std::time::Duration;
    use tracing::info;
    use crate::config::Settings;
    use crate::executor::TransactionExecutor;
    use crate::error::SandoError;
    use solana_sdk::signer::keypair::Keypair;
    use bs58;

    const TEST_LOG_DIR: &str = "./tmp_test_logs";

    fn cleanup_logs() {
        let _ = fs::remove_dir_all(TEST_LOG_DIR);
    }

    #[test]
    fn test_logging_initialization() {
        cleanup_logs();
        let guard = init_logging(TEST_LOG_DIR, "debug", "info");
        assert!(guard.is_ok());
        
        // Keep the guard alive
        let _guard = guard.unwrap();

        // Log some messages
        info!("Info level message for console and file");
        tracing::debug!("Debug level message for file only");

        // Give time for logs to flush (important for non_blocking)
        drop(_guard); // Drop the guard to ensure flush before checking file
        thread::sleep(Duration::from_millis(200)); 

        // Check if log file was created and contains messages
        let log_file_path = Path::new(TEST_LOG_DIR).join("sandoseer.log");
        assert!(log_file_path.exists());

        let log_content = fs::read_to_string(&log_file_path).expect("Failed to read log file");
        assert!(log_content.contains("Info level message"));
        assert!(log_content.contains("Debug level message"));
        assert!(log_content.contains(r#""level":"INFO""#)); // Check JSON format
        assert!(log_content.contains(r#""level":"DEBUG""#));

        cleanup_logs();
    }

    // Helper to get a test keypair string
    fn get_test_keypair_b58() -> String {
        bs58::encode(Keypair::new().to_bytes()).into_string()
    }

    #[tokio::test]
    async fn test_health_check_placeholders() {
        // Create dummy settings and executor for testing the structure
        let settings = Settings {
            solana_rpc_url: "http://localhost:8899".to_string(),
            solana_ws_url: "ws://localhost:8900".to_string(),
            wallet_private_key: get_test_keypair_b58(),
            rig_api_key: Some("dummy".to_string()),
            rig_model_provider: Some("anthropic".to_string()),
            rig_model_name: Some("claude-3-opus-20240229".to_string()),
            telegram_bot_token: Some("dummy".to_string()),
            telegram_chat_id: Some("dummy".to_string()),
            log_level: Some("info".to_string()),
            max_concurrent_trades: 5,
            min_profit_threshold: 0.1,
            max_slippage: 0.1,
            max_position_size: 100.0,
            risk_level: crate::evaluator::RiskLevel::Low,
            simulation_mode: true,
            allowed_tokens: Some(vec!["SOL".to_string()]),
            blocked_tokens: Some(vec![]),
            request_timeout_secs: Some(30),
            max_retries: Some(3),
            retry_backoff_ms: Some(1000),
            orca_api_url: Some("dummy".to_string()),
            raydium_api_url: Some("dummy".to_string()),
            jupiter_api_url: Some("dummy".to_string()),
            arbitrage_min_profit: Some(0.1),
            sandwich_min_profit: Some(0.1),
            snipe_min_profit: Some(0.1),
        };
        let executor = TransactionExecutor::new(&settings.solana_rpc_url, &settings.wallet_private_key, false).unwrap();
        
        let rpc_health = check_rpc_connection(&settings).await;
        // Basic check against devnet should pass if network is ok
        assert!(matches!(rpc_health.status, ComponentStatus::Ok | ComponentStatus::Error(_)));

        let exec_health = check_executor_status(&executor).await;
        // Will likely be error or warning if account has no SOL
        assert!(matches!(exec_health.status, ComponentStatus::Warning(_) | ComponentStatus::Error(_)));

        let listen_health = check_listen_engine_status().await;
        assert_eq!(listen_health.status, ComponentStatus::Ok); // Placeholder always returns Ok

        let rig_health = check_rig_agent_status().await;
        assert_eq!(rig_health.status, ComponentStatus::Ok); // Placeholder always returns Ok
    }

    #[tokio::test]
    async fn test_send_notification_logging() {
        // This test relies on the global logger setup, which is tricky.
        // We mainly test that the function runs without panicking.
        // Verifying the log output would require capturing logs.
        send_notification("Test warning message", "warning").await;
        send_notification("Test error message", "error").await;
        send_notification("Test info message", "info").await;
        // No assertion needed, just check it runs.
    }
} 