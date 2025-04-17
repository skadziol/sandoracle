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
use crate::evaluator::{OpportunityEvaluator, ExecutionThresholds, MevOpportunity};
use crate::executor::{TransactionExecutor, ExecutionService};
use crate::rig_agent::RigAgent;
use crate::market_data::{MarketDataCollector, MarketData};
use tracing::{info, error, warn};
use dotenv::dotenv;
use tokio::signal;
use std::sync::Arc;
use std::collections::HashMap;
use crate::config::RiskLevel as ConfigRiskLevel;
use crate::evaluator::RiskLevel as EvalRiskLevel;
use crate::monitoring::OPPORTUNITY_LOGGER;
use crate::evaluator::arbitrage::ArbitrageEvaluator;
use crate::evaluator::sandwich::SandwichEvaluator;
use crate::evaluator::token_snipe::TokenSnipeEvaluator;

// Add this struct definition for HealthCheckService
pub struct HealthCheckService {
    rpc_urls: Vec<String>,
    notification_urls: Vec<String>,
}

impl HealthCheckService {
    pub fn new(rpc_urls: Vec<String>, notification_urls: Vec<String>) -> Self {
        Self {
            rpc_urls,
            notification_urls,
        }
    }
}

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
    
    // Initialize the RIG Agent
    info!("Initializing RigAgent...");
    let rig_agent = RigAgent::from_env()
        .map_err(|e| SandoError::DependencyError(format!("Failed to create RigAgent: {}", e)))?;
    
    // Wrap the rig_agent in an Arc for safe sharing between threads
    let rig_agent_arc = Arc::new(rig_agent);
    
    // Initialize Market Data Collector
    info!("Initializing MarketDataCollector...");
    let price_cache = Arc::new(tokio::sync::RwLock::new(HashMap::new()));
    let historical_data = Arc::new(tokio::sync::RwLock::new(HashMap::new()));
    let market_data_collector = MarketDataCollector::new(price_cache.clone(), historical_data.clone());
    
    // Initialize with Jupiter API if URL is available
    let market_data_collector = if let Some(jupiter_api_url) = &settings.jupiter_api_url {
        match market_data_collector.clone().with_jupiter(jupiter_api_url) {
            Ok(collector) => {
                info!("Jupiter API integration enabled for real-time price data");
                collector
            },
            Err(e) => {
                error!(error = %e, "Failed to initialize Jupiter client, using simulation data");
                market_data_collector
            }
        }
    } else {
        info!("No Jupiter API URL configured, using simulation data");
        market_data_collector
    };
    
    // Clone the market data collector for use in the update loop
    let market_data_collector_for_update = market_data_collector.clone();
    
    // Wrap market data collector in an Arc for safe sharing
    let market_data_collector_arc = Arc::new(market_data_collector);
    
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
    
    // Register strategy evaluators
    info!("Registering strategy evaluators...");
    
    // Register ArbitrageEvaluator
    let arbitrage_evaluator = ArbitrageEvaluator::new();
    evaluator.register_evaluator(Box::new(arbitrage_evaluator));
    
    // Register SandwichEvaluator
    let sandwich_evaluator = SandwichEvaluator::new();
    evaluator.register_evaluator(Box::new(sandwich_evaluator));
    
    // Register TokenSnipeEvaluator
    let token_snipe_evaluator = TokenSnipeEvaluator::new();
    evaluator.register_evaluator(Box::new(token_snipe_evaluator));
    
    // Set the execution service on the evaluator
    evaluator.set_strategy_executor(execution_service_arc.clone()).await;
    
    // Set the RIG Agent on the evaluator for AI-powered decision making
    evaluator.set_rig_agent(rig_agent_arc.clone()).await;

    // Wrap the evaluator in an Arc for safe sharing between threads
    let evaluator_arc = Arc::new(evaluator);
    
    // Initialize ListenBot (this will internally init listen-engine)
    let (mut listen_bot, listen_bot_cmd_tx) = ListenBot::from_settings(&settings).await?;
    
    // Set the evaluator on the ListenBot - this establishes the data flow from listen-bot to evaluator
    listen_bot.set_evaluator(evaluator_arc.clone());

    // Define token mints for monitoring
    let token_mints = std::collections::HashMap::from([
        ("SOL", "So11111111111111111111111111111111111111112"),  // SOL
        ("USDC", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"), // USDC
        ("ETH", "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"),  // ETH (Wormhole)
        ("BTC", "3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh"),  // BTC (Wormhole)
        ("BONK", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"), // BONK
        ("JUP", "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"),   // JUP
    ]);

    // Configure ListenBot filtering based on token mints from market data
    let token_mint_list: Vec<String> = token_mints.iter()
        .map(|(_, mint)| mint.to_string())
        .collect();
    
    // Prepare program IDs to monitor
    let program_ids_to_monitor = vec![
        // Jupiter v4 and v6
        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4".to_string(),
        "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB".to_string(),
        // Orca
        "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc".to_string(), // Whirlpools
        "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP".to_string(), // v2
        // Raydium
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8".to_string(), // SwapV2
        "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK".to_string(), // CLMM
    ];
    
    // Set minimum transaction value based on risk level
    let min_tx_value = match settings.risk_level {
        crate::config::RiskLevel::Low => 5_000_000,    // 5 SOL in lamports for low risk
        crate::config::RiskLevel::Medium => 1_000_000, // 1 SOL in lamports for medium risk
        crate::config::RiskLevel::High => 500_000,     // 0.5 SOL in lamports for high risk
    };
    
    // Configure the block filter
    listen_bot.configure_block_filter(
        Some(program_ids_to_monitor),
        Some(token_mint_list),
        Some(min_tx_value),
        Some(1), // At least 1 transaction per block
    );
    
    // If in debug mode, run block filter calibration
    if std::env::var("RUST_LOG").map(|v| v.to_lowercase().contains("debug")).unwrap_or(false) {
        info!("Running block filter calibration for optimized filtering...");
        match listen_bot.calibrate_block_filter().await {
            Ok(_) => info!("Block filter calibration complete"),
            Err(e) => warn!(error = %e, "Block filter calibration failed, using default settings"),
        }
    }

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

    // Start the MarketDataCollector update loop in a separate task
    // This task will periodically update market data for decision making
    let market_data_update_handle = tokio::spawn(async move {
        // Update market data every 15 seconds
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(15));
        
        // Sample token list - in production this would be dynamic based on current transactions
        let tokens = vec!["SOL", "USDC", "ETH", "BTC", "BONK", "JUP"];
        let token_mints = HashMap::from([
            ("SOL", "So11111111111111111111111111111111111111112"),  // SOL
            ("USDC", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"), // USDC
            ("ETH", "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"),  // ETH (Wormhole)
            ("BTC", "3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh"),  // BTC (Wormhole)
            ("BONK", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"), // BONK
            ("JUP", "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"),   // JUP
        ]);
        let usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";
        
        loop {
            interval.tick().await;
            
            // Try to get real market data for each token
            for &token in &tokens {
                if let Some(token_mint) = token_mints.get(token) {
                    let market_data = match market_data_collector_for_update.fetch_real_market_data(token_mint, usdc_mint).await {
                        Ok(Some(price)) => {
                            // We have real price data from Jupiter
                            info!(token = token, price = price, "Fetched real-time price");
                            
                            // Create complete market data (some fields are estimated)
                            MarketData {
                                price,
                                // Placeholder market cap and other metrics - in production would fetch from CoinGecko or similar
                                market_cap: price * match token {
                                    "SOL" => 550_000_000.0,
                                    "USDC" => 35_000_000_000.0,
                                    "ETH" => 120_000_000.0,
                                    "BTC" => 19_400_000.0,
                                    "BONK" => 500_000_000_000.0,
                                    "JUP" => 1_000_000_000.0,
                                    _ => 1_000_000.0,
                                },
                                volume_24h: price * match token {
                                    "SOL" => 2_000_000_000.0,
                                    "USDC" => 5_000_000_000.0,
                                    "ETH" => 10_000_000_000.0,
                                    "BTC" => 30_000_000_000.0,
                                    "BONK" => 500_000_000.0,
                                    "JUP" => 100_000_000.0,
                                    _ => 1_000_000.0,
                                },
                                // Calculate price change using history
                                price_change_24h: 0.0, // Will be updated with historical data
                                liquidity: price * match token {
                                    "SOL" => 500_000_000.0,
                                    "USDC" => 1_000_000_000.0,
                                    "ETH" => 2_000_000_000.0,
                                    "BTC" => 5_000_000_000.0,
                                    "BONK" => 100_000_000.0,
                                    "JUP" => 50_000_000.0,
                                    _ => 100_000.0,
                                },
                                volatility: 0.01, // Default placeholder
                                last_update: chrono::Utc::now().timestamp() as u64,
                            }
                        },
                        Ok(None) | Err(_) => {
                            // Fallback to simulated data
                            let now = chrono::Utc::now().timestamp() as f64;
                            let price = match token {
                                "SOL" => 150.0 + (now.sin() * 5.0), // Simple price simulation
                                "USDC" => 1.0,
                                "ETH" => 3500.0 + (now.cos() * 50.0),
                                "BTC" => 65000.0 + (now.sin() * 1000.0),
                                "BONK" => 0.00001 + (now.cos() * 0.000001),
                                "JUP" => 1.2 + (now.sin() * 0.1),
                                _ => 1.0,
                            };
                            
                            // Update price in the shared cache
                            let mut cache = price_cache.write().await;
                            cache.insert(token.to_string(), price);
                            
                            // Update historical data (keep last 24 data points)
                            let mut history = historical_data.write().await;
                            let token_history = history.entry(token.to_string()).or_insert_with(Vec::new);
                            token_history.push(price);
                            
                            // Keep only the last 24 data points (assuming 15-second intervals = 6 minutes)
                            if token_history.len() > 24 {
                                token_history.remove(0);
                            }
                            
                            // Create simulated market data
                            MarketData {
                                price,
                                market_cap: price * match token {
                                    "SOL" => 550_000_000.0, // Rough supply estimates
                                    "USDC" => 35_000_000_000.0,
                                    "ETH" => 120_000_000.0,
                                    "BTC" => 19_400_000.0,
                                    "BONK" => 500_000_000_000.0,
                                    "JUP" => 1_000_000_000.0,
                                    _ => 1_000_000.0,
                                },
                                volume_24h: price * match token {
                                    "SOL" => 2_000_000_000.0,
                                    "USDC" => 5_000_000_000.0, 
                                    "ETH" => 10_000_000_000.0,
                                    "BTC" => 30_000_000_000.0,
                                    "BONK" => 500_000_000.0,
                                    "JUP" => 100_000_000.0,
                                    _ => 1_000_000.0,
                                },
                                // Calculate simple price change using history
                                price_change_24h: if token_history.len() >= 2 {
                                    let oldest = token_history.first().unwrap();
                                    ((price - oldest) / oldest) * 100.0
                                } else {
                                    0.0
                                },
                                liquidity: price * match token {
                                    "SOL" => 500_000_000.0,
                                    "USDC" => 1_000_000_000.0,
                                    "ETH" => 2_000_000_000.0,
                                    "BTC" => 5_000_000_000.0, 
                                    "BONK" => 100_000_000.0,
                                    "JUP" => 50_000_000.0,
                                    _ => 100_000.0,
                                },
                                volatility: if token_history.len() >= 3 {
                                    // Simple volatility calculation
                                    token_history.windows(2)
                                        .map(|w| ((w[1] - w[0]) / w[0]).abs())
                                        .sum::<f64>() / (token_history.len() - 1) as f64
                                } else {
                                    0.01 // Default low volatility
                                },
                                last_update: chrono::Utc::now().timestamp() as u64,
                            }
                        }
                    };
                    
                    if let Err(e) = market_data_collector_for_update.update_market_data(token, market_data).await {
                        error!(token = token, error = ?e, "Failed to update market data");
                    }
                }
            }
            
            info!("Market data updated for {} tokens", tokens.len());
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

    // Initialize services
    let health_check_service = Arc::new(HealthCheckService::new(
        vec![settings.solana_rpc_url.clone()],
        vec![],
    ));

    // Start opportunity stats logger for monitoring MEV opportunities and trades
    let logs_dir = "logs".to_string();
    std::fs::create_dir_all(&logs_dir).expect("Failed to create logs directory");
    tokio::spawn(monitoring::start_opportunity_stats_logger(logs_dir));
    
    info!("Started opportunity monitoring service");

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
