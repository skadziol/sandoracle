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
    EncodedTransaction,  // Needed for matching on transaction type
    TransactionDetails,
    UiConfirmedBlock,
};
use solana_sdk::transaction::VersionedTransaction;
use tokio::task; // Import for spawn_blocking
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _}; // Add Engine trait
use std::time::Duration; // Add this for retry delay
use std::sync::atomic::{AtomicU64, Ordering};

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

        // Define which DEXes to monitor (example)
        let dexes_to_monitor = vec![DexName::Jupiter, DexName::Raydium, DexName::Orca];

        // Get the evaluator if it exists
        let evaluator = self.evaluator.clone();
        let mut cmd_rx = self.cmd_rx;

        // Move core_engine into the async block
        let core_engine = Arc::clone(&self.core_engine);
        
        // Capture the RPC URL from config for logging in the async block
        let rpc_url = self.core_config.rpc_url.clone();

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

            // Log the RPC URL being used (safely without showing full URL)
            let safe_rpc_url = rpc_url.replace("https://", "").replace("http://", "");
            let safe_rpc_parts: Vec<&str> = safe_rpc_url.split('.').collect();
            let masked_url = if safe_rpc_parts.len() > 1 {
                format!("{}.***.{}", safe_rpc_parts[0], safe_rpc_parts.last().unwrap_or(&""))
            } else {
                "RPC URL".to_string()
            };
            info!("Using Solana RPC: {}", masked_url);

            // Get the RPC client from the core engine
            let rpc_client = core_engine.rpc_client();

            let block_config = RpcBlockConfig {
                encoding: Some(UiTransactionEncoding::Base64),
                transaction_details: Some(solana_transaction_status::TransactionDetails::Signatures),
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
                                    5, // Increase from 3 to 5 retries
                                    500, // Increase initial delay from 100 to 500ms
                                ).await {
                                    Ok(Some(block)) => {
                                        if let Some(transactions) = block.transactions {
                                            debug!(slot, num_txs = transactions.len(), "Successfully fetched block with {} transaction signatures", transactions.len());
                                            
                                            // --- Process only transactions with signatures --- 
                                            for (tx_idx, tx_with_meta) in transactions.into_iter().enumerate() {
                                                // Skip transactions without meta
                                                if tx_with_meta.meta.is_none() {
                                                    continue;
                                                }
                                                
                                                // We're only supporting JSON transactions for now
                                                // This is safe because we're using TransactionDetails::Signatures
                                                // which should encode transactions in JSON format
                                                let signature_str = if let EncodedTransaction::Json(json_tx) = &tx_with_meta.transaction {
                                                    json_tx.signatures.get(0).cloned()
                                                } else {
                                                    // Log other encoding types that we encounter
                                                    debug!(
                                                        tx_idx = tx_idx, 
                                                        slot,
                                                        encoding = ?tx_with_meta.transaction, 
                                                        "Unsupported transaction encoding"
                                                    );
                                                    None
                                                };
                                                
                                                if let Some(sig_str) = signature_str {
                                                    let signature = match Signature::from_str(&sig_str) {
                                                        Ok(sig) => sig,
                                                        Err(_) => continue,
                                                    };
                                                    
                                                    debug!(tx_idx = tx_idx, %signature, slot, "Processing transaction signature");
                                                    
                                                    // --- Pass to Evaluator --- 
                                                    if let Some(evaluator) = evaluator.clone() {
                                                        // Use signatures for initial filtering - a simpler version
                                                        // In a production system, you might fetch full transaction
                                                        // details only for promising candidates
                                                        let eval_data = serde_json::json!({ 
                                                            "event_type": "solana_transaction_signature",
                                                            "signature": signature.to_string(),
                                                            "slot": slot,
                                                            "tx_idx": tx_idx,
                                                        });

                                                        debug!(data = ?eval_data, "Evaluating transaction signature");
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
                                                
                                                // Note: The original code for parsing full transaction bytes is now bypassed
                                                // as we're using the signature-based approach. If you need full transaction
                                                // details, you would fetch them via get_transaction after initial filtering.
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

    /// Send a transaction event to the SandoEngine for processing through the full data flow
    pub async fn send_to_engine(&self, event: TransactionEvent) -> Result<()> {
        debug!(event_signature = ?event.signature, "Sending transaction event to SandoEngine");
        
        // First check if we have an evaluator set (required for data flow)
        if let Some(evaluator) = &self.evaluator {
            // Create JSON representation of event for evaluation
            let event_json = match serde_json::to_value(&event) {
                Ok(json) => json,
                Err(e) => {
                    error!(error = %e, "Failed to serialize event to JSON");
                    return Err(SandoError::DataProcessing(
                        format!("Failed to serialize event: {}", e)
                    ));
                }
            };
            
            // Send to the evaluator which is our primary data flow path
            match evaluator.evaluate_opportunity(event_json).await {
                Ok(opportunities) => {
                    if !opportunities.is_empty() {
                        debug!(
                            opportunity_count = opportunities.len(),
                            "Found opportunities from transaction event"
                        );
                        
                        // Process the first opportunity if available
                        // In a complete implementation this would likely happen in the engine
                        if let Some(mut opportunity) = opportunities.into_iter().next() {
                            match evaluator.process_mev_opportunity(&mut opportunity).await {
                                Ok(Some(signature)) => {
                                    info!(
                                        tx_signature = %signature,
                                        strategy = ?opportunity.strategy,
                                        estimated_profit = opportunity.estimated_profit,
                                        "Successfully executed opportunity"
                                    );
                                }
                                Ok(None) => {
                                    debug!("Opportunity evaluation completed but no execution performed");
                                }
                                Err(e) => {
                                    error!(error = %e, "Failed to process opportunity");
                                }
                            }
                        }
                    } else {
                        debug!("No opportunities found from transaction event");
                    }
                }
                Err(e) => {
                    error!(error = %e, "Failed to evaluate opportunity");
                    return Err(SandoError::from(e));
                }
            }
        } else {
            warn!("Cannot process event - no evaluator set on ListenBot");
            return Err(SandoError::DataProcessing(
                "No evaluator set on ListenBot".to_string()
            ));
        }
        
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