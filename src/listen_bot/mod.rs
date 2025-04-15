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
    UiConfirmedBlock,
};
use solana_sdk::transaction::VersionedTransaction;
use tokio::task; // Import for spawn_blocking
use base64::prelude::*; // Use base64 prelude for BASE64_STANDARD
use base64::Engine; // Import Engine trait for decode method
use std::time::Duration; // Add this for retry delay
use std::sync::atomic::{AtomicU64, Ordering};
use crate::listen_bot::stream::{TransactionStream, StreamConfig};
use crate::listen_bot::transaction::{TransactionMonitor, DexTransactionParser};
use crate::listen_bot::dex::JupiterParser;

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

// Add these static counters 
static TX_DESERIALIZATION_ERRORS: AtomicU64 = AtomicU64::new(0);
static TX_DECODE_ERRORS: AtomicU64 = AtomicU64::new(0);
static LAST_ERROR_LOG_TIME: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
enum DecodedTransaction {
    Legacy(solana_sdk::transaction::Transaction),
    Versioned(solana_sdk::transaction::VersionedTransaction)
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

    /// Fetch a block with retry logic
    async fn fetch_block_with_retry(
        rpc_client: Arc<RpcClient>,  // Use Arc<RpcClient> instead of RpcClient
        slot: u64, 
        config: RpcBlockConfig,
        max_retries: u32,
        initial_delay_ms: u64,
    ) -> Result<Option<UiConfirmedBlock>> {
        let mut retries = 0;
        let mut delay_ms = initial_delay_ms;
        
        loop {
            // Clone the config since we'll need it in multiple iterations
            let config_clone = config.clone();
            let client = rpc_client.clone();
            
            // Try to fetch the block
            match task::spawn_blocking(move || {
                client.get_block_with_config(slot, config_clone)
            }).await {
                Ok(Ok(block)) => return Ok(Some(block)),
                Ok(Err(err)) => {
                    // Check if we've reached max retries
                    if retries >= max_retries {
                        error!(
                            slot, 
                            error = %err, 
                            max_retries,
                            "Failed to fetch block after maximum retries"
                        );
                        return Err(SandoError::SolanaRpc(
                            format!("Failed to fetch block slot={} after {} retries: {}", 
                            slot, max_retries, err)
                        ));
                    }
                    
                    // Check if error indicates block is not available (which we should retry)
                    let err_str = err.to_string();
                    let is_block_not_available = err_str.contains("Block not available") || 
                                               err_str.contains("-32004");
                    
                    if !is_block_not_available {
                        // For other errors, don't retry
                        error!(slot, error = %err, "Failed to fetch block with non-retriable error");
                        return Err(SandoError::SolanaRpc(
                            format!("Failed to fetch block with non-retriable error: {}", err)
                        ));
                    }
                    
                    // Log retry attempt
                    warn!(
                        slot, 
                        retry_count = retries + 1, 
                        max_retries,
                        delay_ms,
                        "Block not available, retrying after delay"
                    );
                    
                    // Sleep with exponential backoff
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    
                    // Increase retry counter and delay with exponential backoff
                    retries += 1;
                    delay_ms = std::cmp::min(delay_ms * 2, 2000); // Cap at 2 seconds
                }
                Err(e) => {
                    error!(slot, error = %e, "Block fetching task failed");
                    return Err(SandoError::InternalError(
                        format!("Block fetching task failed: {}", e)
                    ));
                }
            }
        }
    }

    /// Starts the ListenBot's main loop using listen-core
    pub async fn start(self) -> Result<()> {
        info!("Starting ListenBot (listen-core based)...");

        // --- Start new TransactionStream for DEX log streaming ---
        let stream_config = StreamConfig::default(); // You may want to build from settings
        let monitor = TransactionMonitor::default(); // Or build from config
        let parsers: Vec<Arc<dyn DexTransactionParser>> = vec![
            Arc::new(JupiterParser),
        ];
        let (transaction_stream, _rx) = TransactionStream::new(stream_config, monitor, parsers);
        tokio::spawn(async move {
            if let Err(e) = transaction_stream.start().await {
                error!("TransactionStream failed: {:?}", e);
            }
        });
        // --- End TransactionStream integration ---

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
                                
                                // Use the retry-enabled block fetching method
                                match Self::fetch_block_with_retry(
                                    rpc_client.clone(), // rpc_client is already an Arc<RpcClient>
                                    slot, 
                                    block_config.clone(),
                                    3, // Max 3 retries
                                    100, // Start with 100ms delay
                                ).await {
                                    Ok(Some(block)) => {
                                        if let Some(transactions) = block.transactions {
                                            debug!(slot, num_txs = transactions.len(), "Successfully fetched block");
                                            
                                            // --- Iterate & Process Transactions --- 
                                            for tx_with_meta in transactions {
                                                // Filter out failed transactions
                                                if let Some(ref meta) = tx_with_meta.meta {
                                                    // We'll check transaction success another way
                                                    // The err field format changed in newer Solana versions
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
                                                        // Don't log every JSON transaction, just increment counter
                                                        TX_DECODE_ERRORS.fetch_add(1, Ordering::Relaxed);
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
                                                        // Don't log every account transaction, just increment counter
                                                        TX_DECODE_ERRORS.fetch_add(1, Ordering::Relaxed);
                                                        continue;
                                                    }
                                                };

                                                let decoded_tx = match decoded_tx {
                                                    Some(bytes) => {
                                                        // Try versioned transaction first (more common in newer blocks)
                                                        if let Ok(versioned_tx) = bincode::deserialize::<solana_sdk::transaction::VersionedTransaction>(&bytes) {
                                                            DecodedTransaction::Versioned(versioned_tx)
                                                        } 
                                                        // Fall back to legacy format
                                                        else if let Ok(legacy_tx) = bincode::deserialize::<solana_sdk::transaction::Transaction>(&bytes) {
                                                            DecodedTransaction::Legacy(legacy_tx)
                                                        } else {
                                                            // Count error but only log periodically
                                                            let error_count = TX_DESERIALIZATION_ERRORS.fetch_add(1, Ordering::Relaxed);
                                                            
                                                            // Only log every 100th error or once per minute
                                                            let now = std::time::SystemTime::now()
                                                                .duration_since(std::time::UNIX_EPOCH)
                                                                .unwrap_or_default()
                                                                .as_secs();
                                                            
                                                            let last_log_time = LAST_ERROR_LOG_TIME.load(Ordering::Relaxed);
                                                            
                                                            // If it's been a minute since last log or we've hit the 100th error
                                                            if now > last_log_time + 60 || error_count % 100 == 0 {
                                                                LAST_ERROR_LOG_TIME.store(now, Ordering::Relaxed);
                                                                
                                                                // Log a summary of errors instead of individual ones
                                                                warn!(
                                                                    slot, 
                                                                    deserialization_errors = error_count,
                                                                    decode_errors = TX_DECODE_ERRORS.load(Ordering::Relaxed),
                                                                    "Transaction processing errors summary"
                                                                );
                                                            }
                                                            continue;
                                                        }
                                                    },
                                                    None => {
                                                        // Count the error but don't log every one
                                                        TX_DECODE_ERRORS.fetch_add(1, Ordering::Relaxed);
                                                        continue;
                                                    }
                                                };

                                                // Process based on transaction version
                                                let (signature, signer) = match decoded_tx {
                                                    DecodedTransaction::Legacy(tx) => {
                                                        // Get signature from decoded transaction
                                                        let signature = match tx.signatures.first() {
                                                            Some(sig) => sig.to_string(),
                                                            None => {
                                                                TX_DECODE_ERRORS.fetch_add(1, Ordering::Relaxed);
                                                                continue;
                                                            }
                                                        };

                                                        // Get the signer (fee payer)
                                                        let signer = match tx.message.account_keys.get(0) {
                                                            Some(pubkey) => pubkey.to_string(),
                                                            None => {
                                                                TX_DECODE_ERRORS.fetch_add(1, Ordering::Relaxed);
                                                                continue;
                                                            }
                                                        };

                                                        (signature, signer)
                                                    },
                                                    DecodedTransaction::Versioned(tx) => {
                                                        // Get signature from versioned transaction
                                                        let signature = match tx.signatures.first() {
                                                            Some(sig) => sig.to_string(),
                                                            None => {
                                                                TX_DECODE_ERRORS.fetch_add(1, Ordering::Relaxed);
                                                                continue;
                                                            }
                                                        };

                                                        // Get the signer (fee payer) - the first account in the static account keys
                                                        let signer = match tx.message.static_account_keys().first() {
                                                            Some(pubkey) => pubkey.to_string(),
                                                            None => {
                                                                TX_DECODE_ERRORS.fetch_add(1, Ordering::Relaxed);
                                                                continue;
                                                            }
                                                        };
                                                        
                                                        (signature, signer)
                                                    }
                                                };

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
                                        } else {
                                            debug!(slot, "Block has no transactions");
                                        }
                                    }
                                    Ok(None) => {
                                        warn!(slot, "Block not available after all retries, skipping");
                                    }
                                    Err(e) => {
                                        error!(slot, error = %e, "Failed to fetch block");
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