use listen_core::{
    ListenEngine, ListenEngineConfig, // Core engine components
    model::tx::Transaction as ListenTransaction, // Transaction model from listen-core
    router::dexes::DexName, // DEX enum if needed for configuration
};

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
use futures::stream::StreamExt;
use solana_client::rpc_client::RpcClient;
use solana_client::rpc_config::{RpcBlockConfig, RpcTransactionConfig};
use solana_sdk::commitment_config::CommitmentConfig;
use solana_transaction_status::{
    UiTransactionEncoding,
    TransactionBinaryEncoding,
    EncodedTransaction,
    TransactionDetails,
    option_serializer::OptionSerializer,
};
use solana_sdk::transaction::VersionedTransaction;
use tokio::task; // Import for spawn_blocking
use base64::prelude::*; // Use base64 prelude for BASE64_STANDARD
use base64::Engine; // Import Engine trait for decode method

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
    /// The underlying listen-core instance
    core_engine: Arc<ListenEngine>,
    /// Configuration
    core_config: ListenEngineConfig,
    /// Transaction event receiver (from listen-core)
    core_event_rx: Option<mpsc::Receiver<ListenTransaction>>,
    /// Transaction event sender (for internal use or forwarding) - Might be repurposed or removed
    event_tx: mpsc::Sender<TransactionEvent>,
    /// Command receiver (for shutdown signal)
    cmd_rx: mpsc::Receiver<ListenBotCommand>,
    /// Opportunity evaluator for MEV detection
    evaluator: Option<Arc<OpportunityEvaluator>>,
}

impl ListenBot {
    /// Creates a new ListenBot instance using settings from the environment.
    pub async fn from_settings(settings: &Settings) -> Result<(Self, mpsc::Sender<ListenBotCommand>)> {
        info!("Initializing ListenBot using listen-core...");

        // Create ListenEngineConfig from global Settings
        let core_config = ListenEngineConfig {
            rpc_url: settings.solana_rpc_url.clone(),
            ws_url: Some(settings.solana_ws_url.clone()), // Use ws_url from settings
            commitment: settings.commitment.clone().unwrap_or_else(|| "confirmed".to_string()), // Use commitment from settings or default
        };

        // Create the listen-core engine
        let core_engine = ListenEngine::new(core_config.clone())
            .map_err(|e| SandoError::DependencyError(format!("Failed to create listen-core Engine: {}", e)))?;
        let core_engine_arc = Arc::new(core_engine);

        let (cmd_tx, cmd_rx) = mpsc::channel::<ListenBotCommand>(1);
        let (internal_event_tx, _internal_event_rx) = mpsc::channel::<TransactionEvent>(1000);

        info!("ListenBot (listen-core based) initialized successfully.");
        Ok((
            Self {
                core_engine: core_engine_arc,
                core_config,
                core_event_rx: None, // Stream receiver will be set in start()
                event_tx: internal_event_tx,
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

    /// Starts the ListenBot's main loop using listen-core
    pub async fn start(self) -> Result<()> {
        info!("Starting ListenBot (listen-core based)...");

        // Define which DEXes to monitor (example)
        let dexes_to_monitor = vec![DexName::Jupiter, DexName::Raydium, DexName::Orca];

        // Get the evaluator if it exists
        let evaluator = self.evaluator.clone();
        let mut cmd_rx = self.cmd_rx;

        // Move core_engine into the async block
        let core_engine = Arc::clone(&self.core_engine);

        let event_processor_handle = tokio::spawn(async move {
            info!("ListenBot event processor started...");

            // Get the slot stream from listen-core engine inside the async block
            let slot_stream = match core_engine.stream_dex_swaps(dexes_to_monitor).await {
                Ok(stream) => stream,
                Err(e) => {
                    error!(error = %e, "Failed to get stream from listen-core");
                    return;
                }
            };

            // Get the RPC client from the core engine
            let rpc_client = core_engine.rpc_client();

            let block_config = RpcBlockConfig {
                encoding: Some(UiTransactionEncoding::Base64),
                transaction_details: Some(solana_transaction_status::TransactionDetails::Full),
                rewards: Some(false),
                commitment: Some(CommitmentConfig::confirmed()),
                max_supported_transaction_version: Some(0),
            };

            let mut slot_stream = slot_stream;

            loop {
                tokio::select! {
                    // Check for shutdown command
                    Some(command) = cmd_rx.recv() => {
                        match command {
                            ListenBotCommand::Shutdown => {
                                info!("Shutdown command received. Stopping ListenBot event processor...");
                                break;
                            }
                        }
                    }
                    // Process next slot from the listen-core stream
                    maybe_slot = slot_stream.next() => {
                        match maybe_slot {
                            Some(slot) => {
                                debug!(slot, "Received slot, fetching block...");
                                
                                // Clone client for blocking task
                                let client = rpc_client.clone(); 
                                let config = block_config.clone();
                                
                                // --- Fetch Block using spawn_blocking --- 
                                match task::spawn_blocking(move || client.get_block_with_config(slot, config)).await {
                                    Ok(Ok(block)) => {
                                        if let Some(transactions) = block.transactions {
                                            debug!(slot, num_txs = transactions.len(), "Successfully fetched block");
                                            
                                            // --- Iterate & Process Transactions --- 
                                            for tx_with_meta in transactions {
                                                // Filter out failed transactions
                                                if let Some(ref meta) = tx_with_meta.meta {
                                                    if meta.err.is_some() {
                                                        continue;
                                                    }
                                                } else {
                                                    continue;
                                                }

                                                // Get logs early to avoid move issues
                                                let logs = tx_with_meta.meta
                                                    .as_ref()
                                                    .and_then(|m| match &m.log_messages {
                                                        OptionSerializer::Some(logs) => Some(logs.clone()),
                                                        _ => None,
                                                    })
                                                    .unwrap_or_default();

                                                // Decode transaction first
                                                let decoded_tx = match &tx_with_meta.transaction {
                                                    EncodedTransaction::Json(_) => {
                                                        warn!(slot, "JSON encoded transaction not supported");
                                                        continue;
                                                    },
                                                    EncodedTransaction::Binary(data, encoding) => {
                                                        match encoding {
                                                            TransactionBinaryEncoding::Base58 => bs58::decode(data).into_vec().ok(),
                                                            TransactionBinaryEncoding::Base64 => base64::prelude::BASE64_STANDARD.decode(data).ok(),
                                                        }
                                                    },
                                                    EncodedTransaction::LegacyBinary(data) => {
                                                        // Legacy binary is base-58 encoded
                                                        bs58::decode(data).into_vec().ok()
                                                    },
                                                    EncodedTransaction::Accounts(_) => {
                                                        warn!(slot, "Account-based transaction encoding not supported");
                                                        continue;
                                                    }
                                                };

                                                let decoded_tx = match decoded_tx {
                                                    Some(bytes) => match bincode::deserialize::<solana_sdk::transaction::Transaction>(&bytes) {
                                                        Ok(tx) => tx,
                                                        Err(e) => {
                                                            warn!(slot, error = %e, "Failed to deserialize transaction");
                                                            continue;
                                                        }
                                                    },
                                                    None => {
                                                        warn!(slot, "Failed to decode transaction data");
                                                        continue;
                                                    }
                                                };

                                                // Get signature from decoded transaction
                                                let signature = match &decoded_tx {
                                                    solana_sdk::transaction::Transaction { signatures, .. } => {
                                                        if let Some(sig) = signatures.first() {
                                                            sig.to_string()
                                                        } else {
                                                            warn!(slot, "Transaction missing signature");
                                                            continue;
                                                        }
                                                    }
                                                };

                                                // Process based on transaction version
                                                match decoded_tx {
                                                    solana_sdk::transaction::Transaction { message, .. } => {
                                                        // Get the signer (fee payer)
                                                        let signer = message.account_keys.get(0).map(|pk| pk.to_string()).unwrap_or_default();
                                                        if signer.is_empty() {
                                                            warn!(slot, %signature, "Transaction missing signer (account_keys[0])?");
                                                            continue;
                                                        }

                                                        let block_time = block.block_time.unwrap_or_else(|| chrono::Utc::now().timestamp());
                                                        
                                                        debug!(%signature, %signer, slot, "Processing transaction");

                                                        if let Some(evaluator) = evaluator.clone() {
                                                            // Construct Eval Data
                                                            let eval_data = serde_json::json!({
                                                                "event_type": "solana_transaction",
                                                                "signature": signature,
                                                                "slot": slot,
                                                                "block_time": block_time,
                                                                "signer": signer,
                                                                "logs": logs,
                                                            });

                                                            // Evaluate Opportunity
                                                            debug!(data = ?eval_data, "Evaluating potential opportunity...");
                                                            match evaluator.evaluate_opportunity(eval_data).await {
                                                                Ok(opportunities) => {
                                                                    if !opportunities.is_empty() {
                                                                        info!(count = opportunities.len(), %signature, "Found potential MEV opportunities");
                                                                        for (idx, mut opportunity) in opportunities.into_iter().enumerate() {
                                                                            match evaluator.process_mev_opportunity(&mut opportunity).await {
                                                                                Ok(Some(exec_sig)) => {
                                                                                    info!(idx = idx, strategy = ?opportunity.strategy, signature = %exec_sig, "Successfully executed MEV opportunity");
                                                                                },
                                                                                Ok(None) => {
                                                                                    info!(idx = idx, strategy = ?opportunity.strategy, decision = ?opportunity.decision, "Decided not to execute MEV opportunity");
                                                                                },
                                                                                Err(e) => {
                                                                                    error!(idx = idx, strategy = ?opportunity.strategy, error = %e, "Error executing MEV opportunity");
                                                                                }
                                                                            }
                                                                        }
                                                                    } 
                                                                }
                                                                Err(e) => {
                                                                    error!(%signature, error = %e, "Error evaluating opportunity");
                                                                }
                                                            }
                                                        } else {
                                                            warn!("No evaluator set for ListenBot, cannot evaluate transaction.");
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            debug!(slot, "Block has no transactions");
                                        }
                                    }
                                    Ok(Err(e)) => {
                                        error!(slot, error = %e, "Failed to fetch block");
                                    }
                                    Err(e) => {
                                        error!(slot, error = %e, "Block fetching task failed");
                                    }
                                }
                            }
                            None => {
                                info!("Listen-core slot stream ended. Exiting loop.");
                                break;
                            }
                        }
                    }
                }
            }
            info!("ListenBot event processor stopped.");
        });

        // Wait for the event processor task to complete
        if let Err(e) = event_processor_handle.await {
            error!(error = ?e, "ListenBot event processor task failed or panicked");
            return Err(SandoError::InternalError(format!("Event processor panicked: {:?}", e)));
        }

        info!("ListenBot shutting down.");
        Ok(())
    }

    /// Stops the ListenBot and its underlying engine.
    pub async fn stop(&mut self) -> Result<()> {
        // TODO: Need a way to signal the event_processor_handle task to stop.
        // The self.cmd_rx is moved into the task.
        // Maybe return the cmd_tx from from_settings and use it in main.rs?
        warn!("ListenBot::stop needs refactoring to correctly signal the processing loop.");
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