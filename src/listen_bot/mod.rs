use listen_engine::engine::Engine;
use sandoracle_types::events::TransactionEvent as EngineTransactionEvent;
use listen_engine::server::state::EngineMessage;
use listen_engine::engine::pipeline::{Pipeline, Status};

use crate::config::Settings;
use crate::error::{Result, SandoError};
use crate::evaluator::{OpportunityEvaluator, MevOpportunity};
use tokio::sync::{mpsc, oneshot};
use tracing::{info, error, warn, debug};
use std::sync::Arc;
use chrono;
use uuid;
use std::collections::HashMap;
use std::str::FromStr;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Signature;

// Import PriceUpdate from listen-engine
use listen_engine::redis::subscriber::PriceUpdate;

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
    /// Transaction event sender (for internal use or forwarding) - Might be repurposed or removed
    event_tx: mpsc::Sender<TransactionEvent>,
    /// Price Update receiver (from listen-engine)
    price_update_rx: mpsc::Receiver<PriceUpdate>,
    /// Command receiver (for shutdown signal)
    cmd_rx: mpsc::Receiver<ListenBotCommand>,
    /// Opportunity evaluator for MEV detection
    evaluator: Option<Arc<OpportunityEvaluator>>,
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
        let (internal_event_tx, _internal_event_rx) = mpsc::channel::<TransactionEvent>(1000); // Keep for now, might remove later
        // let (_engine_event_tx, engine_event_rx) = mpsc::channel::<EngineTransactionEvent>(1000); // REMOVE THIS - Engine uses PriceUpdate

        // Get the engine and the correct PriceUpdate receiver
        let (engine, price_rx) = Engine::from_env().await
            .map_err(|e| SandoError::DependencyError(format!("Failed to create listen-engine Engine: {}", e)))?;

        let engine_arc = Arc::new(engine);

        info!("ListenBot initialized successfully.");
        Ok((
            Self {
                engine: engine_arc,
                config: bot_config,
                event_tx: internal_event_tx, // Keep for now
                price_update_rx: price_rx, // USE the receiver from the engine
                cmd_rx,
                evaluator: None,
            },
            cmd_tx,
        ))
    }

    /// Set the opportunity evaluator for MEV detection
    pub fn set_evaluator(&mut self, evaluator: Arc<OpportunityEvaluator>) {
        info!("Setting OpportunityEvaluator for ListenBot");
        self.evaluator = Some(evaluator);
    }

    /// Starts the ListenBot's main loop and the underlying engine.
    pub async fn start(mut self) -> Result<()> {
        info!("Starting ListenBot...");

        // self.create_monitoring_pipeline().await?; // This seems engine-specific, remove from here?
        // info!("Listen Engine pipeline configured."); // This seems engine-specific

        let engine_to_start = self.engine.clone();
        
        // Create channels for engine command communication (if needed by engine.run)
        // let (_price_tx, price_rx) = mpsc::channel(1000); // REMOVE THIS - we already have the receiver
        let (cmd_tx, cmd_rx) = mpsc::channel(1000); // Engine run needs a command channel

        // Create a dummy pipeline for testing? - This seems like internal engine logic, remove from here
        // let test_pipeline = Pipeline { ... };
        // Send test pipeline to engine? - Remove from here
        // let (response_tx, _response_rx) = oneshot::channel();
        // let _ = cmd_tx.send(EngineMessage::AddPipeline { ... }).await;

        // Note: engine.run needs the PriceUpdate receiver, but it gets it from Engine::from_env internally via its subscriber.
        // We just need to pass the command channel.
        let engine_handle = tokio::spawn(async move {
            info!("Starting listen-engine...");
            // Pass the command receiver to engine run loop
            // The PriceUpdate receiver is handled internally by the engine via its Redis subscriber
            if let Err(e) = Engine::run(engine_to_start, mpsc::channel(1).1, cmd_rx).await { // Pass dummy price receiver, engine uses its own
                error!(error = %e, "Listen Engine failed to start or encountered an error");
            } else {
                info!("Listen Engine stopped.");
            }
        });

        // Get the evaluator if it exists
        let evaluator = self.evaluator.clone();

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
                    // Listen on the price_update_rx channel now
                    Some(price_update) = self.price_update_rx.recv() => {
                        debug!(asset = %price_update.pubkey, price = price_update.price, slot = price_update.slot, "ListenBot received PriceUpdate from channel");
                        // Rename function to reflect it processes price updates
                        match Self::process_price_update(price_update, evaluator.clone()).await {
                            Ok(Some(processed_event)) => {
                                // Still send TransactionEvent for now, but might change this
                                if let Err(e) = self.event_tx.send(processed_event).await {
                                    error!("Failed to send processed event internally: {}", e);
                                }
                            }
                            Ok(None) => { /* Filtered or no action needed */ }
                            Err(e) => {
                                error!("Error processing price update: {}", e);
                            }
                        }
                    }
                    else => {
                        info!("ListenBot price update channel closed. Exiting loop.");
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

    /// Processes a price update received from the listen-engine's channel.
    async fn process_price_update(
        price_update: PriceUpdate, 
        evaluator: Option<Arc<OpportunityEvaluator>>
    ) -> Result<Option<TransactionEvent>> {
        info!(asset = %price_update.pubkey, price = price_update.price, slot = price_update.slot, "Processing price update");

        // TODO: Decide if this price update warrants creating an "Opportunity"
        // For now, let's always try to evaluate if an evaluator exists.

        // If we have an evaluator, evaluate based on the price update
        if let Some(evaluator) = evaluator {
            debug!(asset = %price_update.pubkey, "Evaluating price update for MEV opportunities");
            
            // Construct data for the evaluator based on the PriceUpdate
            // This structure needs to align with what OpportunityEvaluator expects
            let eval_data = serde_json::json!({
                "event_type": "price_update",
                "asset_mint": price_update.pubkey,
                "price": price_update.price,
                "slot": price_update.slot,
                "timestamp": chrono::Utc::now().timestamp(),
                // Potentially add other market context here if available
            });

            // Evaluate for MEV opportunities based on price change
            match evaluator.evaluate_opportunity(eval_data).await {
                Ok(opportunities) => {
                    if !opportunities.is_empty() {
                        info!(
                            count = opportunities.len(),
                            asset = %price_update.pubkey,
                            "Found potential MEV opportunities based on price update"
                        );
                        
                        // Process each opportunity (same logic as before)
                        for (idx, mut opportunity) in opportunities.into_iter().enumerate() {
                            info!(
                                idx = idx,
                                strategy = ?opportunity.strategy,
                                profit = opportunity.estimated_profit,
                                confidence = opportunity.confidence,
                                risk = ?opportunity.risk_level,
                                "Processing MEV opportunity triggered by price update"
                            );
                            
                            match evaluator.process_mev_opportunity(&mut opportunity).await {
                                Ok(Some(signature)) => {
                                    info!(
                                        idx = idx,
                                        strategy = ?opportunity.strategy,
                                        signature = %signature,
                                        "Successfully executed MEV opportunity (from price update)"
                                    );
                                },
                                Ok(None) => {
                                    info!(
                                        idx = idx,
                                        strategy = ?opportunity.strategy,
                                        decision = ?opportunity.decision,
                                        "Decided not to execute MEV opportunity (from price update)"
                                    );
                                },
                                Err(e) => {
                                    error!(
                                        idx = idx,
                                        strategy = ?opportunity.strategy,
                                        error = %e,
                                        "Error executing MEV opportunity (from price update)"
                                    );
                                }
                            }
                        }
                    } else {
                        debug!(asset = %price_update.pubkey, "No opportunities found for this price update");
                    }
                }
                Err(e) => {
                    error!(asset = %price_update.pubkey, error = %e, "Error evaluating opportunity for price update");
                }
            }
        } else {
            warn!("No evaluator set for ListenBot, cannot evaluate price update.");
        }

        // For now, return None as we are not converting PriceUpdate to TransactionEvent
        // This might change based on how downstream components use the event_tx channel.
        Ok(None) 
    }

    /// Stops the ListenBot and its underlying engine.
    pub async fn stop(&mut self) -> Result<()> {
        info!("Sending shutdown command to ListenBot event processor...");
        // The actual cmd_tx was moved into the task, we need a way to access it or redesign shutdown
        // For now, this won't work as intended.
        // if let Err(e) = self.cmd_rx.send(ListenBotCommand::Shutdown).await { // This seems wrong, should send on cmd_tx
        //     error!("Failed to send shutdown command: {}", e);
        //     return Err(SandoError::InternalError(format!("Failed to send shutdown: {}", e)));
        // }
        warn!("ListenBot::stop currently cannot signal the running task correctly.");
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