use listen_engine::engine::Engine;
use sandoracle_types::events::TransactionEvent as EngineTransactionEvent;
use listen_engine::server::state::EngineMessage;
use listen_engine::engine::pipeline::{Pipeline, Status};

use crate::config::Settings;
use crate::error::{Result, SandoError};
use tokio::sync::{mpsc, oneshot};
use tracing::{info, error, warn};
use std::sync::Arc;
use chrono;
use uuid;
use std::collections::HashMap;
use std::str::FromStr;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Signature;

mod config;
mod types;
mod dex;
mod transaction;
mod stream;

pub use config::ListenBotConfig;
pub use transaction::TransactionEvent;

/// Commands for controlling the ListenBot
#[derive(Debug)]
pub enum ListenBotCommand {
    Shutdown,
}

/// Main ListenBot struct that manages transaction monitoring
pub struct ListenBot {
    /// The underlying listen-engine instance
    engine: Arc<Engine>,
    /// Configuration
    config: ListenBotConfig,
    /// Transaction event sender (for internal use or forwarding)
    event_tx: mpsc::Sender<TransactionEvent>,
    /// Transaction event receiver (from listen-engine)
    event_rx: mpsc::Receiver<EngineTransactionEvent>,
    /// Command receiver (for shutdown signal)
    cmd_rx: mpsc::Receiver<ListenBotCommand>,
}

impl ListenBot {
    /// Creates a new ListenBot instance using settings from the environment.
    pub async fn from_settings(settings: &Settings) -> Result<(Self, mpsc::Sender<ListenBotCommand>)> {
        info!("Initializing ListenBot from settings...");
        // Create ListenBotConfig from global Settings
        let bot_config = ListenBotConfig {
            rpc_url: settings.solana_rpc_url.clone(),
            ws_url: settings.solana_ws_url.clone(),
            filter: Default::default(),
            max_concurrent_requests: settings.max_concurrent_trades,
            request_timeout: settings.request_timeout_secs.unwrap_or(30),
            max_retries: settings.max_retries.unwrap_or(3),
            retry_backoff_ms: settings.retry_backoff_ms.unwrap_or(1000),
        };

        let (cmd_tx, cmd_rx) = mpsc::channel::<ListenBotCommand>(1);
        let (internal_event_tx, _internal_event_rx) = mpsc::channel::<TransactionEvent>(1000);
        let (_engine_event_tx, engine_event_rx) = mpsc::channel::<EngineTransactionEvent>(1000);

        let (engine, _price_rx) = Engine::from_env().await
            .map_err(|e| SandoError::DependencyError(format!("Failed to create listen-engine Engine: {}", e)))?;

        let engine_arc = Arc::new(engine);

        info!("ListenBot initialized successfully.");
        Ok((
            Self {
                engine: engine_arc,
                config: bot_config,
                event_tx: internal_event_tx,
                event_rx: engine_event_rx,
                cmd_rx,
            },
            cmd_tx,
        ))
    }

    /// Starts the ListenBot's main loop and the underlying engine.
    pub async fn start(mut self) -> Result<()> {
        info!("Starting ListenBot...");

        self.create_monitoring_pipeline().await?;
        info!("Listen Engine pipeline configured.");

        let engine_to_start = self.engine.clone();
        
        // Create channels for engine communication
        let (_price_tx, price_rx) = mpsc::channel(1000);
        let (cmd_tx, cmd_rx) = mpsc::channel(1000);

        // Create a dummy pipeline for testing
        let test_pipeline = Pipeline {
            id: uuid::Uuid::new_v4(),
            user_id: "test".to_string(),
            wallet_address: None,
            pubkey: None,
            current_steps: Vec::new(),
            steps: HashMap::new(),
            status: Status::Pending,
            created_at: chrono::Utc::now(),
        };

        // Send test pipeline to engine
        let (response_tx, _response_rx) = oneshot::channel();
        let _ = cmd_tx.send(EngineMessage::AddPipeline {
            pipeline: test_pipeline,
            response_tx,
        }).await;

        let engine_handle = tokio::spawn(async move {
            info!("Starting listen-engine...");
            if let Err(e) = Engine::run(engine_to_start, price_rx, cmd_rx).await {
                error!(error = %e, "Listen Engine failed to start or encountered an error");
            } else {
                info!("Listen Engine stopped.");
            }
        });

        let event_processor_handle = tokio::spawn(async move {
            info!("ListenBot event processor started.");
            loop {
                tokio::select! {
                    Some(command) = self.cmd_rx.recv() => {
                        match command {
                            ListenBotCommand::Shutdown => {
                                info!("Shutdown command received. Stopping ListenBot event processor...");
                                break;
                            }
                        }
                    }
                    Some(engine_event) = self.event_rx.recv() => {
                        match Self::process_engine_event(engine_event).await {
                            Ok(Some(processed_event)) => {
                                if let Err(e) = self.event_tx.send(processed_event).await {
                                    error!("Failed to send processed event internally: {}", e);
                                }
                            }
                            Ok(None) => { /* Filtered */ }
                            Err(e) => {
                                error!("Error processing engine event: {}", e);
                            }
                        }
                    }
                    else => {
                        info!("ListenBot event channel closed. Exiting loop.");
                        break;
                    }
                }
            }
            info!("ListenBot event processor stopped.");
        });

        tokio::select! {
            _ = engine_handle => {
                warn!("Listen Engine task completed or failed.");
            },
            _ = event_processor_handle => {
                 info!("ListenBot event processor task completed.");
            }
        }

        info!("ListenBot shutting down.");
        Ok(())
    }

    /// Processes an event received from the listen-engine.
    async fn process_engine_event(event: EngineTransactionEvent) -> Result<Option<TransactionEvent>> {
        info!("Processing engine event");

        // Convert the engine event to our local TransactionEvent format
        let processed_event = TransactionEvent {
            signature: Signature::from_str(&event.transaction_hash)
                .map_err(|e| SandoError::InternalError(format!("Invalid signature format: {}", e)))?,
            program_id: Pubkey::from_str(&event.details) // Assuming details contains the program ID
                .map_err(|e| SandoError::InternalError(format!("Invalid program ID format: {}", e)))?,
            token_mint: None, // We don't have this info from the engine event
            amount: None, // We don't have this info from the engine event
            success: event.status == "Executed", // Convert status string to boolean
        };

        Ok(Some(processed_event))
    }

    /// Creates the monitoring pipeline using the listen-engine configuration API.
    async fn create_monitoring_pipeline(&self) -> Result<()> {
        info!("Configuring listen-engine pipeline...");
        if !self.config.filter.program_ids.is_empty() {
            let program_ids = self.config.filter.program_ids.iter().cloned().collect::<Vec<_>>();
            info!(count = program_ids.len(), "Adding program ID filter to engine pipeline");
            warn!("Engine pipeline configuration (filtering) is currently a placeholder.");
        }
        Ok(())
    }

    /// Stops the ListenBot and its underlying engine.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping ListenBot...");

        // Shutdown the engine gracefully - Note: shutdown() might not return Result
        self.engine.shutdown().await; // Removed error handling based on compiler output
        // Removed: if let Err(e) = self.engine.shutdown().await {
        //     error!("Error during listen-engine shutdown: {}", e);
        //     return Err(SandoError::DependencyError(format!("Engine shutdown failed: {}", e)));
        // }
        info!("Listen Engine shutdown requested.");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, timeout};
    use std::time::Duration;
    use crate::config::Settings; // Import Settings

    // Helper to create default settings for tests
    fn test_settings() -> Settings {
        // Set minimal required env vars for Settings::from_env() to work in tests
        std::env::set_var("SOLANA_RPC_URL", "http://127.0.0.1:8899"); // Use local validator URL for tests
        std::env::set_var("SOLANA_WS_URL", "ws://127.0.0.1:8900");
        std::env::set_var("WALLET_PRIVATE_KEY", bs58::encode([0u8; 64].to_vec()).into_string()); // Dummy key
        std::env::set_var("ANTHROPIC_API_KEY", "dummy-anthropic-key"); // Dummy key
        Settings::from_env().expect("Failed to create test settings from env")
    }

    #[tokio::test]
    #[ignore] // Ignored because it requires a running Solana validator and engine interaction
    async fn test_listenbot_lifecycle() -> Result<()> {
        let settings = test_settings(); // Use helper
        let (mut bot, cmd_tx) = ListenBot::from_settings(&settings).await?;

        // Start the bot in a separate task
        let bot_handle = tokio::spawn(async move {
            bot.start().await
        });

        // Allow some time for the bot to start
        sleep(Duration::from_secs(2)).await;

        // Send shutdown command
        info!("Sending shutdown command to ListenBot...");
        cmd_tx.send(ListenBotCommand::Shutdown).await.expect("Failed to send shutdown command");

        // Wait for the bot task to finish with a timeout
        match timeout(Duration::from_secs(5), bot_handle).await {
            Ok(Ok(Ok(_))) => {
                info!("ListenBot task completed successfully after shutdown command.");
            }
            Ok(Ok(Err(e))) => {
                panic!("ListenBot task returned an error: {}", e);
            }
            Ok(Err(join_err)) => {
                panic!("ListenBot task panicked or was cancelled: {}", join_err);
            }
            Err(_) => {
                 panic!("ListenBot task did not complete within timeout after shutdown command.");
            }
        }

        Ok(())
    }
} 