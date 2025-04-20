use listen_core::{
    ListenEngine, ListenEngineConfig, // Core engine components
    model::tx::Transaction as ListenTransaction, // Transaction model from listen-core
    router::dexes::DexName, // DEX enum if needed for configuration
};

use crate::config::Settings;
use crate::error::{Result, SandoError};
use crate::evaluator::{OpportunityEvaluator, MevOpportunity, ExecutionDecision};
use crate::market_data::MarketDataCollector;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, error, warn, debug, trace};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use uuid;
use std::collections::HashMap;
use std::str::FromStr;
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Signature;
use futures::stream::StreamExt;
use solana_client::rpc_client::RpcClient;
use solana_client::rpc_config::{RpcBlockConfig, RpcTransactionConfig};
use solana_client::rpc_response::{RpcLogsResponse, RpcKeyedAccount};
use solana_rpc_client_api::filter::RpcFilterType;
use solana_client::nonblocking::pubsub_client::PubsubClient;
use solana_rpc_client_api::config::{RpcTransactionLogsConfig, RpcTransactionLogsFilter};
use solana_sdk::commitment_config::CommitmentConfig;
use solana_transaction_status::{
    UiTransactionEncoding,
    EncodedTransaction,  // Needed for matching on transaction type
    TransactionDetails,
    UiConfirmedBlock,
    UiMessage,  // Add this import for pattern matching on message types
    UiParsedMessage,  // Add this for accessing the parsed message fields
    UiInstruction,    // Add this for working with instruction types
    UiParsedInstruction, // Add UiParsedInstruction for proper pattern matching
    EncodedConfirmedTransactionWithStatusMeta,
};
use solana_transaction_status::parse_accounts::ParsedAccount;  // Import from correct module
use tokio::task; // Import for spawn_blocking
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _}; // Add Engine trait
use std::time::Duration; // Add this for retry delay
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashSet;
use serde_json::{json, Value};
use tokio::time::{sleep, timeout};
use anyhow::anyhow;
use regex::Regex;
use tokio_stream::StreamExt as TokioStreamExt;
use futures_util;
use solana_sdk::{message::Message, instruction::CompiledInstruction, transaction::VersionedTransaction}; // Add Message, CompiledInstruction
use bincode; // Add bincode
use bs58; // Add bs58 if handling LegacyBinary
use crate::types::{DecodedTransactionDetails, try_decode_transaction};

// Set to true to relax filtering criteria for debugging (more transactions will pass)
const DEBUG_RELAX_FILTERS: bool = false;

mod config;
mod types;
mod dex;
mod transaction;
mod stream;

pub use config::ListenBotConfig;

/// Transaction event from ListenBot
#[derive(Debug, Clone, serde::Serialize)]
pub struct TransactionEvent {
    /// Block slot
    pub slot: u64,
    /// Transaction signature / hash
    pub signature: String,
    /// Transaction data (parsed)
    pub data: Value,
    /// Transaction program IDs (for filtering)
    pub program_ids: Vec<String>,
    /// Transaction account keys (for filtering)
    pub account_keys: Vec<String>,
    /// Transaction timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Transaction value in lamports
    pub transaction_value: u64,
    /// Transaction instructions count
    pub instructions_count: u32,
    /// Unique ID for this event
    pub id: String,
    /// Whether the transaction is pending (mempool) or confirmed
    pub is_pending: bool,
}

/// Filter criteria for blocks and transactions
#[derive(Debug, Clone)]
pub struct BlockFilter {
    /// Minimum transaction value (in lamports)
    pub min_transaction_value: u64,
    /// Monitored program IDs (DEXs, etc.)
    pub monitored_program_ids: HashSet<String>,
    /// Monitored token mints
    pub monitored_token_mints: HashSet<String>,
    /// Minimum number of transactions required to process a block
    pub min_transactions: usize,
}

impl Default for BlockFilter {
    fn default() -> Self {
        Self {
            min_transaction_value: 1_000_000, // 0.001 SOL minimum
            monitored_program_ids: HashSet::new(),
            monitored_token_mints: HashSet::new(),
            min_transactions: 1, // Default to processing all blocks with at least 1 transaction
        }
    }
}

impl BlockFilter {
    /// Adds solana program IDs to monitor
    pub fn with_monitored_programs(mut self, program_ids: Vec<String>) -> Self {
        for id in program_ids {
            self.monitored_program_ids.insert(id);
        }
        self
    }
    
    /// Adds token mints to monitor
    pub fn with_monitored_tokens(mut self, token_mints: Vec<String>) -> Self {
        for mint in token_mints {
            self.monitored_token_mints.insert(mint);
        }
        self
    }
    
    /// Sets the minimum transaction value
    pub fn with_min_transaction_value(mut self, value: u64) -> Self {
        self.min_transaction_value = value;
        self
    }
    
    /// Sets the minimum number of transactions required to process a block
    pub fn with_min_transactions(mut self, count: usize) -> Self {
        self.min_transactions = count;
        self
    }
    
    /// Checks whether a block should be processed based on transaction count
    pub fn should_process_block(&self, tx_count: usize) -> bool {
        // If DEBUG_RELAX_FILTERS is true, we'll skip the filtering for debugging
        if DEBUG_RELAX_FILTERS {
            debug!("DEBUG_RELAX_FILTERS enabled: bypassing block transaction count filter");
            return true;
        }
        
        // Check if the block has enough transactions to process
        if tx_count < self.min_transactions {
            debug!(
                tx_count,
                min_transactions = self.min_transactions,
                "Block filtered: insufficient transaction count"
            );
            return false;
        }
        
        debug!("Block passed filter: has sufficient transaction count");
        true
    }
    
    /// Checks whether a transaction should be processed
    pub fn should_process_transaction(&self, program_ids: &[String], account_keys: &[String]) -> bool {
        // If DEBUG_RELAX_FILTERS is true, we'll skip the filtering for debugging
        if DEBUG_RELAX_FILTERS {
            debug!("DEBUG_RELAX_FILTERS enabled: bypassing normal filter criteria");
            return true;
        }
        
        // Skip if no program IDs match our monitored list
        let has_monitored_program = program_ids.iter()
            .any(|id| self.monitored_program_ids.contains(id));
            
        if !has_monitored_program {
            debug!(
                program_ids_count = program_ids.len(),
                monitored_programs_count = self.monitored_program_ids.len(),
                "Transaction filtered: no monitored program IDs found"
            );
            return false;
        }
        
        // Skip if no account keys match our monitored token mints
        let has_monitored_token = account_keys.iter()
            .any(|key| self.monitored_token_mints.contains(key));
            
        if !has_monitored_token {
            debug!(
                account_keys_count = account_keys.len(),
                monitored_tokens_count = self.monitored_token_mints.len(),
                "Transaction filtered: no monitored token mints found"
            );
            return false;
        }
        
        debug!("Transaction passed filter: has monitored program and token mint");
        true
    }
}

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
    /// Block filtering settings
    block_filter: BlockFilter,
    /// Flag indicating whether to use mempool or confirmed blocks
    use_mempool: bool,
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
        
        // Create a customized block filter based on settings
        let block_filter = {
            let mut filter = BlockFilter::default();
            
            // If risk level is set to low, increase the minimum transaction value
            if let crate::config::RiskLevel::Low = settings.risk_level {
                filter.min_transaction_value = 5_000_000; // 5 SOL in lamports for low risk
            }
            
            // Medium risk level uses the default
            
            // If risk level is high, lower minimum transaction value
            if let crate::config::RiskLevel::High = settings.risk_level {
                filter.min_transaction_value = 500_000; // 0.5 SOL in lamports for high risk
            }
            
            // Configure based on other settings if needed
            if settings.min_profit_threshold > 100.0 {
                filter.min_transactions = 3; // Only process blocks with 3+ transactions
            }
            
            filter
        };

        info!("ListenBot (listen-core based) initialized successfully with block filtering.");
        Ok((
            Self {
                core_engine: core_engine_arc,
                core_config,
                core_event_rx: None, // Stream receiver will be set in start()
                event_tx: internal_event_tx,
                cmd_rx,
                evaluator: None,
                block_filter,
                use_mempool: false,
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

        // Print diagnostic information about the filter settings
        info!(
            min_transactions = self.block_filter.min_transactions,
            min_transaction_value = self.block_filter.min_transaction_value,
            monitored_programs = self.block_filter.monitored_program_ids.len(),
            monitored_tokens = self.block_filter.monitored_token_mints.len(),
            "ListenBot filter settings"
        );
        
        // Log whether DEBUG_RELAX_FILTERS is enabled
        if DEBUG_RELAX_FILTERS {
            warn!("DEBUG_RELAX_FILTERS is enabled - normal filtering criteria are bypassed!");
        }

        // Check if evaluator is set
        if self.evaluator.is_some() {
            info!("Opportunity evaluator is properly connected");
        } else {
            warn!("No opportunity evaluator set - transactions will be filtered but not evaluated");
        }

        // Define which DEXes to monitor (example)
        let dexes_to_monitor = vec![DexName::Jupiter, DexName::Raydium, DexName::Orca];
        info!("Monitoring DEXes: {:?}", dexes_to_monitor);

        // Get the evaluator if it exists
        let evaluator = self.evaluator.clone();
        let mut cmd_rx = self.cmd_rx;

        // Move core_engine into the async block
        let core_engine = Arc::clone(&self.core_engine);
        
        // Capture the RPC URL from config for logging in the async block
        let rpc_url = self.core_config.rpc_url.clone();
        
        // Clone the block filter for use in the async block
        let block_filter = self.block_filter.clone();

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
                encoding: Some(UiTransactionEncoding::Json),
                transaction_details: Some(solana_transaction_status::TransactionDetails::Full),
                rewards: Some(false),
                commitment: Some(CommitmentConfig::confirmed()),
                max_supported_transaction_version: Some(0),
            };

            let mut slot_stream = slot_stream;
            
            // Track statistics for block filtering
            let mut total_blocks_seen: u64 = 0;
            let mut blocks_processed: u64 = 0;
            let mut transactions_seen: u64 = 0;
            let mut transactions_processed: u64 = 0;
            let mut last_stats_time = std::time::Instant::now();

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
                    maybe_slot = futures::stream::StreamExt::next(&mut slot_stream) => {
                        match maybe_slot {
                            Some(slot) => {
                                debug!(slot, "Received slot, fetching block...");
                                total_blocks_seen += 1;
                                
                                // Use the retry-enabled block fetching method
                                match Self::fetch_block_with_retry(
                                    rpc_client.clone(), // rpc_client is already an Arc<RpcClient>
                                    slot, 
                                    block_config.clone(),
                                    5, // Increase from 3 to 5 retries
                                    500, // Increase initial delay from 100 to 500ms
                                ).await {
                                    Ok(Some(block)) => {
                                        // Add diagnostic logging for block structure
                                        debug!(
                                            slot,
                                            block_has_transactions = block.transactions.is_some(),
                                            transactions_len = block.transactions.as_ref().map(|txs| txs.len()).unwrap_or(0),
                                            block_hash = %block.blockhash,
                                            parent_slot = block.parent_slot,
                                            "Block structure overview"
                                        );
                                        
                                        if let Some(transactions) = block.transactions {
                                            let tx_count = transactions.len();
                                            transactions_seen += tx_count as u64;
                                            
                                            // Add diagnostic logging for the first transaction if available
                                            if tx_count > 0 {
                                                if let Some(first_tx) = transactions.get(0) {
                                                    // Log raw transaction structure to investigate the format
                                                    debug!(
                                                        ?first_tx, // Log the entire transaction structure 
                                                        "Raw transaction structure for debugging"
                                                    );
                                                    
                                                    debug!(
                                                        slot,
                                                        tx_idx = 0,
                                                        tx_has_meta = first_tx.meta.is_some(),
                                                        tx_type = match &first_tx.transaction {
                                                            EncodedTransaction::Json(json_tx) => {
                                                                // Log raw JSON transaction structure
                                                                debug!(?json_tx, "JSON transaction structure");
                                                                "Json"
                                                            },
                                                            EncodedTransaction::Binary(_, _) => "Binary",
                                                            EncodedTransaction::LegacyBinary(_) => "LegacyBinary",
                                                            EncodedTransaction::Accounts(_) => "Accounts",
                                                        },
                                                        "First transaction structure"
                                                    );
                                                    
                                                    // Log more details if it's a JSON transaction
                                                    if let EncodedTransaction::Json(json_tx) = &first_tx.transaction {
                                                        let message_type = match &json_tx.message {
                                                            UiMessage::Parsed(_) => "Parsed",
                                                            UiMessage::Raw(_) => "Raw",
                                                        };
                                                        
                                                        debug!(
                                                            slot,
                                                            tx_idx = 0,
                                                            message_type,
                                                            signature_count = json_tx.signatures.len(),
                                                            "First transaction JSON details"
                                                        );
                                                    }
                                                }
                                            }
                                            
                                            // Apply early block filtering
                                            if !block_filter.should_process_block(tx_count) {
                                                debug!(slot, "Block has insufficient transactions ({}) to process", tx_count);
                                                continue;
                                            }
                                            
                                            debug!(slot, num_txs = tx_count, "Successfully fetched block with {} transaction signatures", tx_count);
                                            blocks_processed += 1;
                                            
                                            // --- Process only transactions with signatures --- 
                                            for (tx_idx, tx_with_meta) in transactions.into_iter().enumerate() {
                                                // Skip transactions without meta
                                                if tx_with_meta.meta.is_none() {
                                                    continue;
                                                }
                                                
                                                // Extract transaction value if available - simplified approach
                                                let transaction_value = if let Some(meta) = &tx_with_meta.meta {
                                                    meta.fee
                                                } else {
                                                    0
                                                };
                                                
                                                // Filter by transaction value
                                                debug!(
                                                    tx_idx = tx_idx,
                                                    %transaction_value,
                                                    min_value = block_filter.min_transaction_value,
                                                    "Transaction value check"
                                                );
                                                
                                                if transaction_value < block_filter.min_transaction_value {
                                                    debug!(tx_idx = tx_idx, %transaction_value, "Skipping low-value transaction");
                                                    continue;
                                                }
                                                
                                                debug!(tx_idx = tx_idx, %transaction_value, slot, "Processing transaction signature");
                                                
                                                // --- Decode Transaction Attempt (Task 4.1) --- 
                                                let mut signature_str = "UNKNOWN_SIG".to_string(); // Declare signature_str *before* match
                                                let mut decoded_details: Option<DecodedTransactionDetails> = None; // Declare decoded_details *before* logic
 
                                                // Match on the EncodedTransaction enum to get the message
                                                let maybe_versioned_tx: Option<VersionedTransaction> = match &tx_with_meta.transaction {
                                                    EncodedTransaction::Json(json_tx) => {
                                                        // Assign signature within arm if needed - Correct for Json variant
                                                        signature_str = json_tx.signatures.get(0).cloned().unwrap_or_else(|| "MISSING_SIG".to_string());
                                                        warn!(target: "listen_bot::start", tx_sig=%signature_str, "Decoding JSON encoded tx in confirmed block is complex, skipping for now.");
                                                        None 
                                                    }
                                                    EncodedTransaction::Binary(encoded_data, encoding) => {
                                                        // Removed premature signature_str assignment here
                                                        let decoded_data = match encoding {
                                                            solana_transaction_status::TransactionBinaryEncoding::Base64 => BASE64_STANDARD.decode(encoded_data).ok(),
                                                            solana_transaction_status::TransactionBinaryEncoding::Base58 => bs58::decode(encoded_data).into_vec().ok(),
                                                            _ => { 
                                                                // Assign unknown signature here if encoding fails
                                                                signature_str = "UNKNOWN_SIG_ENCODING".to_string();
                                                                warn!(target: "listen_bot::start", tx_sig=%signature_str, ?encoding, "Unsupported binary encoding"); 
                                                                None 
                                                            }
                                                        };
                                                        if let Some(ref bytes) = decoded_data {
                                                            match bincode::deserialize::<VersionedTransaction>(bytes) {
                                                                Ok(tx) => {
                                                                    // Assign signature_str *after* successful deserialization
                                                                    signature_str = tx.signatures.get(0).map(|s| s.to_string()).unwrap_or_else(|| "MISSING_SIG_POST_DECODE".to_string());
                                                                    Some(tx)
                                                                }, 
                                                                Err(e) => {
                                                                    // Assign unknown signature here if deserialize fails
                                                                    signature_str = "UNKNOWN_SIG_DESERIALIZE".to_string();
                                                                    warn!(target: "listen_bot::start", tx_sig=%signature_str, error=%e, "Failed to deserialize VersionedTransaction from binary data.");
                                                                    None
                                                                }
                                                            }
                                                        } else { 
                                                            // Assign unknown signature here if decoded_data is None
                                                            signature_str = "UNKNOWN_SIG_NO_DECODE".to_string();
                                                            None 
                                                        }
                                                    }
                                                    EncodedTransaction::LegacyBinary(encoded_data) => {
                                                        // Assign unknown signature here - Legacy handling not implemented
                                                         signature_str = "UNKNOWN_SIG_LEGACY".to_string();
                                                         // ... (legacy handling) ...
                                                        None 
                                                    }
                                                    _ => None, 
                                                };

                                                if let Some(tx) = maybe_versioned_tx {
                                                    // --- Call decoder *inside* the successful extraction --- 
                                                    let message = &tx.message;
                                                    let ixs: Vec<CompiledInstruction> = message.instructions().iter().map(|ix| 
                                                        CompiledInstruction {
                                                            program_id_index: ix.program_id_index,
                                                            accounts: ix.accounts.clone(),
                                                            data: ix.data.clone(),
                                                        }
                                                    ).collect();
                                                    // Assign to outer decoded_details, remove inner `let`
                                                    decoded_details = try_decode_transaction(signature_str.clone(), message, &ixs);
                                                } else if signature_str != "UNKNOWN_SIG" { 
                                                     // ... (warning unchanged) ...
                                                }
                                                // --- End Decode --- 
                                                
                                                // --- Pass to Evaluator --- 
                                                if let Some(evaluator) = evaluator.clone() {
                                                    let eval_data = serde_json::json!({ 
                                                        // ... (other fields) ...
                                                        "decoded_details": decoded_details, // Use outer variable
                                                    });
                                                    // ... (evaluator call) ...
                                                }
                                            }
                                        } else {
                                            debug!(slot, "Block has no transactions");
                                        }
                                        
                                        // Log filter stats periodically
                                        let now = std::time::Instant::now();
                                        if now.duration_since(last_stats_time).as_secs() >= 60 {
                                            info!(
                                                blocks_seen = total_blocks_seen,
                                                blocks_processed = blocks_processed,
                                                blocks_filtered = (total_blocks_seen - blocks_processed),
                                                blocks_efficiency = format!("{:.2}%", (blocks_processed as f64 / total_blocks_seen as f64) * 100.0),
                                                transactions_seen = transactions_seen,
                                                transactions_processed = transactions_processed,
                                                transactions_filtered = (transactions_seen - transactions_processed),
                                                tx_efficiency = format!("{:.2}%", (transactions_processed as f64 / transactions_seen.max(1) as f64) * 100.0),
                                                "Block filter efficiency statistics"
                                            );
                                            last_stats_time = now;
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
                                    debug!(
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

    /// Configure block filtering with custom settings
    pub fn configure_block_filter(&mut self, 
        program_ids: Option<Vec<String>>,
        token_mints: Option<Vec<String>>,
        min_value: Option<u64>,
        min_transactions: Option<usize>
    ) {
        let mut updated = false;
        
        if let Some(program_ids) = program_ids {
            info!("Updating monitored program IDs, new count: {}", program_ids.len());
            
            // Log the specific program IDs we're monitoring (truncated for readability)
            if !program_ids.is_empty() {
                let program_display = program_ids.iter()
                    .take(5)
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let extra = if program_ids.len() > 5 { format!(" and {} more", program_ids.len() - 5) } else { "".to_string() };
                info!("Monitoring programs: {}{}", program_display, extra);
            }
            
            self.block_filter.monitored_program_ids = program_ids.into_iter().collect();
            updated = true;
        }
        
        if let Some(token_mints) = token_mints {
            info!("Updating monitored token mints, new count: {}", token_mints.len());
            
            // Log the specific token mints we're monitoring (truncated for readability)
            if !token_mints.is_empty() {
                let token_display = token_mints.iter()
                    .take(5)
                    .map(|mint| mint.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let extra = if token_mints.len() > 5 { format!(" and {} more", token_mints.len() - 5) } else { "".to_string() };
                info!("Monitoring tokens: {}{}", token_display, extra);
            }
            
            self.block_filter.monitored_token_mints = token_mints.into_iter().collect();
            updated = true;
        }
        
        if let Some(min_value) = min_value {
            info!("Updating minimum transaction value: {}", min_value);
            self.block_filter.min_transaction_value = min_value;
            updated = true;
        }
        
        if let Some(min_transactions) = min_transactions {
            info!("Updating minimum transactions per block: {}", min_transactions);
            self.block_filter.min_transactions = min_transactions;
            updated = true;
        }
        
        if updated {
            info!("Block filter configuration updated");
        }
    }
    
    /// Add specific program IDs to monitor
    pub fn add_monitored_program_ids(&mut self, program_ids: Vec<String>) {
        let count_before = self.block_filter.monitored_program_ids.len();
        for id in program_ids {
            self.block_filter.monitored_program_ids.insert(id);
        }
        let count_after = self.block_filter.monitored_program_ids.len();
        info!("Added {} new program IDs to monitoring list", count_after - count_before);
    }
    
    /// Add specific token mints to monitor
    pub fn add_monitored_token_mints(&mut self, token_mints: Vec<String>) {
        let count_before = self.block_filter.monitored_token_mints.len();
        for mint in token_mints {
            self.block_filter.monitored_token_mints.insert(mint);
        }
        let count_after = self.block_filter.monitored_token_mints.len();
        info!("Added {} new token mints to monitoring list", count_after - count_before);
    }
    
    /// Run a test to determine optimal filtering settings based on current blockchain activity
    pub async fn calibrate_block_filter(&mut self) -> Result<()> {
        info!("Calibrating block filter based on recent blockchain activity...");
        
        // Get RPC client
        let rpc_client = self.core_engine.rpc_client();
        
        // Get recent blocks to analyze
        let recent_blocks_config = RpcBlockConfig {
            encoding: Some(UiTransactionEncoding::Json),
            transaction_details: Some(solana_transaction_status::TransactionDetails::Signatures),
            rewards: Some(false),
            commitment: Some(CommitmentConfig::confirmed()),
            max_supported_transaction_version: Some(0),
        };
        
        // Get latest slot and analyze recent blocks
        let latest_blockhash = match rpc_client.get_latest_blockhash() {
            Ok(blockhash) => blockhash,
            Err(e) => {
                return Err(SandoError::SolanaRpc(format!("Failed to get latest blockhash: {}", e)));
            }
        };
        
        // Get slot for the latest blockhash
        let slot = match rpc_client.get_slot() {
            Ok(slot) => slot,
            Err(e) => {
                return Err(SandoError::SolanaRpc(format!("Failed to get current slot: {}", e)));
            }
        };
        
        // Create a sample of slots to analyze (current and a few earlier ones)
        let mut recent_slots = vec![slot];
        for i in 1..10 {
            if slot > i {
                recent_slots.push(slot - i);
            }
        }
        
        let mut transaction_sizes = Vec::new();
        let mut program_id_counts = HashMap::new();
        let mut token_mint_counts = HashMap::new();
        
        // Analyze blocks
        info!("Analyzing {} recent blocks for filter calibration", recent_slots.len());
        
        for slot in recent_slots.iter().take(10) { // Limit to 10 blocks for calibration
            if let Ok(Some(block)) = Self::fetch_block_with_retry(
                rpc_client.clone(),
                *slot,
                recent_blocks_config.clone(),
                3,
                500,
            ).await {
                if let Some(transactions) = block.transactions {
                    for tx in transactions {
                        if let EncodedTransaction::Json(json_tx) = &tx.transaction {
                            // Collect program IDs
                            if let UiMessage::Parsed(parsed_msg) = &json_tx.message {
                                for ix in &parsed_msg.instructions {
                                    match ix {
                                        UiInstruction::Compiled(compiled_ix) => {
                                            // program_id_index is a u8, not an Option
                                            let program_idx = compiled_ix.program_id_index;
                                            if program_idx < parsed_msg.account_keys.len() as u8 {
                                                let program_id = parsed_msg.account_keys[program_idx as usize].pubkey.clone();
                                                *program_id_counts.entry(program_id).or_insert(0) += 1;
                                            }
                                        },
                                        UiInstruction::Parsed(parsed_ix) => {
                                            // Extract program_id based on the variant
                                            match parsed_ix {
                                                UiParsedInstruction::Parsed(parsed) => {
                                                    // For fully parsed instructions, program_id is in the program field
                                                    *program_id_counts.entry(parsed.program_id.clone()).or_insert(0) += 1;
                                                },
                                                UiParsedInstruction::PartiallyDecoded(partial) => {
                                                    // For partially decoded instructions, program_id is available directly
                                                    *program_id_counts.entry(partial.program_id.to_string()).or_insert(0) += 1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Collect top account keys (might be tokens)
                            if let UiMessage::Parsed(parsed_msg) = &json_tx.message {
                                for account in &parsed_msg.account_keys {
                                    let key = account.pubkey.clone();
                                    *token_mint_counts.entry(key).or_insert(0) += 1;
                                }
                            }
                            
                            // Collect transaction sizes
                            match &tx.transaction {
                                EncodedTransaction::Json(_) => {
                                    if let Some(meta) = &tx.meta {
                                        if let (Some(pre), Some(post)) = (meta.pre_balances.get(0), meta.post_balances.get(0)) {
                                            let value = pre.saturating_sub(*post);
                                            if value > 0 {
                                                transaction_sizes.push(value);
                                            }
                                        }
                                    }
                                },
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        
        // Determine optimal filter values
        if !transaction_sizes.is_empty() {
            transaction_sizes.sort();
            let median_size = transaction_sizes[transaction_sizes.len() / 2];
            let p25_size = transaction_sizes[transaction_sizes.len() / 4];
            
            info!("Transaction size analysis: median={}, p25={}", median_size, p25_size);
            
            // Set min transaction value to 25th percentile to capture most relevant transactions
            self.block_filter.min_transaction_value = p25_size;
        }
        
        // Find top program IDs
        let mut program_ids: Vec<_> = program_id_counts.into_iter().collect();
        program_ids.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Find top token mints
        let mut token_mints: Vec<_> = token_mint_counts.into_iter().collect();
        token_mints.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Add top program IDs
        let top_programs: Vec<_> = program_ids.iter()
            .take(10) // Top 10 program IDs
            .map(|(id, count)| format!("{} (count: {})", id, count))
            .collect();
            
        // Add top token mints
        let top_tokens: Vec<_> = token_mints.iter()
            .take(10) // Top 10 token mints
            .map(|(id, count)| format!("{} (count: {})", id, count))
            .collect();
        
        info!("Calibration complete. Top programs: {:?}", top_programs);
        info!("Calibration complete. Top tokens: {:?}", top_tokens);
        
        // Optionally add the top program IDs and token mints to the filter
        // This is commented out to let the user decide whether to use these results
        /*
        if !program_ids.is_empty() {
            self.add_monitored_program_ids(program_ids.iter().take(10).map(|(id, _)| id.clone()).collect());
        }
        
        if !token_mints.is_empty() {
            self.add_monitored_token_mints(token_mints.iter().take(10).map(|(id, _)| id.clone()).collect());
        }
        */
        
        Ok(())
    }

    /// Listen to the mempool for pending transactions
    pub async fn listen_mempool(&self) -> Result<()> {
        // Get the opportunity evaluator from the service
        let evaluator = self.evaluator.clone()
            .ok_or_else(|| anyhow!("Opportunity evaluator not available"))?;
            
        // Get the RPC client from the service
        let rpc_client = Arc::new(RpcClient::new(self.core_config.rpc_url.clone()));
            
        info!(target: "listen_bot", "Starting mempool listener...");
            
        // Create a websocket connection to the Solana node
        let wsock_url = self.core_config.ws_url.clone().unwrap_or_else(|| {
            self.core_config.rpc_url.replace("http", "ws")
        });
        
        info!(target: "listen_bot", "Connecting to mempool via WebSocket: {}", wsock_url);
        
        // Create a new pubsub client for WebSocket connection
        let pubsub_client = PubsubClient::new(&wsock_url).await?;
        
        // Subscribe to transaction logs notifications
        info!(target: "listen_bot", "Subscribing to mempool transactions via logs");
        
        // Create configuration for logs subscription
        let logs_config = RpcTransactionLogsConfig {
            commitment: Some(CommitmentConfig::confirmed()),
        };
        
        // Subscribe to all transaction logs
        let (mut transaction_stream, _unsubscribe) = pubsub_client
            .logs_subscribe(
                RpcTransactionLogsFilter::All,
                logs_config,
            )
            .await?;
            
        info!(target: "listen_bot", "Successfully subscribed to transaction logs");
        
        // Load the monitored tokens from the block filter
        let monitored_tokens: Vec<String> = self.block_filter.monitored_token_mints.iter().cloned().collect();
        
        // Process transactions as they come in
        while let Some(transaction_info) = StreamExt::next(&mut transaction_stream).await {
            let signature_str = transaction_info.value.signature.clone();
            
            // Base transaction data from logs
            let mut transaction_data = json!({
                "is_pending": true,
                "signature": signature_str,
                "transaction": {
                    "signature": signature_str,
                    "meta": {
                        "logMessages": transaction_info.value.logs,
                        "preBalances": [], // Not available from logsSubscribe
                        "postBalances": [], // Not available from logsSubscribe
                    },
                    // message field needs to be populated by get_transaction_details
                },
                "decoded_details": Option::<DecodedTransactionDetails>::None // Initialize decoded_details
            });
            
            let tx_signature = match Signature::from_str(&signature_str) {
                Ok(sig) => sig,
                Err(e) => {
                    warn!(target: "listen_bot::mempool", error = %e, "Invalid transaction signature from logs, skipping");
                    continue;
                }
            };
            
            // Enhance transaction data with more details if available
            let enhanced_data = match self.get_transaction_details(&rpc_client, &tx_signature).await {
                Ok(Some(mut details_value)) => {
                    let mut decoded_details: Option<DecodedTransactionDetails> = None; // Initialize
                    
                    // --- Decode Transaction Attempt (Task 4.1) --- 
                    match serde_json::from_value::<EncodedConfirmedTransactionWithStatusMeta>(details_value.clone()) {
                        Ok(encoded_tx_with_meta) => {
                            // Match on the inner EncodedTransaction enum
                            match encoded_tx_with_meta.transaction.transaction {
                                EncodedTransaction::Json(json_tx) => {
                                    warn!(target: "listen_bot::mempool", tx_sig=%signature_str, "Decoding JSON encoded tx in mempool is complex, skipping for now.");
                                    // Cannot easily get VersionedTransaction from Json variant here
                                }
                                EncodedTransaction::Binary(encoded_data, encoding) => {
                                    let decoded_data = match encoding {
                                        solana_transaction_status::TransactionBinaryEncoding::Base64 => BASE64_STANDARD.decode(encoded_data).ok(),
                                        solana_transaction_status::TransactionBinaryEncoding::Base58 => bs58::decode(encoded_data).into_vec().ok(),
                                        _ => None,
                                    };
                                    if let Some(bytes) = decoded_data {
                                        if let Ok(tx) = bincode::deserialize::<VersionedTransaction>(&bytes) {
                                            // Successfully got the VersionedTransaction
                                            let message = &tx.message;
                                            let ixs: Vec<CompiledInstruction> = message.instructions().iter().map(|ix| 
                                                CompiledInstruction {
                                                    program_id_index: ix.program_id_index,
                                                    accounts: ix.accounts.clone(),
                                                    data: ix.data.clone(),
                                                }
                                            ).collect();
                                            // Now call the decoder
                                            decoded_details = try_decode_transaction(signature_str.clone(), message, &ixs);
                                        } else {
                                            warn!(target: "listen_bot::mempool", tx_sig=%signature_str, "Could not deserialize VersionedTransaction from binary data.");
                                        }
                                    } else {
                                         warn!(target: "listen_bot::mempool", tx_sig=%signature_str, ?encoding, "Unsupported or failed binary decoding.");
                                    }
                                }
                                EncodedTransaction::LegacyBinary(_) => {
                                    warn!(target: "listen_bot::mempool", tx_sig=%signature_str, "Decoding LegacyBinary encoded tx is not yet supported.");
                                }
                                _ => {
                                     warn!(target: "listen_bot::mempool", tx_sig=%signature_str, "Unsupported EncodedTransaction variant encountered.");
                                }
                            }
                        }
                        Err(_) => {
                             warn!(target: "listen_bot::mempool", tx_sig=%signature_str, "Could not deserialize fetched details into EncodedConfirmedTransactionWithStatusMeta.");
                        }
                    }
                    // --- End Decode --- 

                    // Merge details and add decoded info
                    if let Value::Object(map) = &mut details_value {
                         map.insert("is_pending".to_string(), json!(true));
                         map.insert("decoded_details".to_string(), serde_json::to_value(decoded_details).unwrap_or(Value::Null));
                     } else {
                        // If not an object, just use the base data + decoded (which is None)
                         transaction_data["decoded_details"] = serde_json::to_value(decoded_details).unwrap_or(Value::Null);
                         details_value = transaction_data; // Fallback
                     }
                    details_value
                },
                Ok(None) => transaction_data, // Use base data if get_transaction fails initially
                Err(e) => {
                    warn!(target: "listen_bot::mempool", error = %e, "Failed to get transaction details, using base log data");
                    transaction_data
                }
            };
            
            // Check if this is a DEX interaction
            if !self.is_dex_interaction(&enhanced_data) {
                trace!(target: "listen_bot", sig = transaction_info.value.signature, "Not a DEX interaction, skipping");
                continue;
            }
            
            // Send to evaluator using the evaluate_opportunity method
            match evaluator.evaluate_opportunity(enhanced_data).await {
                Ok(opportunities) if !opportunities.is_empty() => {
                    // Process each opportunity found
                    for mut opportunity in opportunities {
                        info!(
                            target: "listen_bot",
                            sig = transaction_info.value.signature,
                            strategy = ?opportunity.strategy,
                            profit = opportunity.estimated_profit,
                            "Found potential opportunity"
                        );
                        
                        // Process the opportunity
                        if let Err(e) = evaluator.process_mev_opportunity(&mut opportunity).await {
                            warn!(target: "listen_bot", error = %e, "Failed to process opportunity");
                        } else if let Some(decision) = opportunity.decision {
                            info!(
                                target: "listen_bot",
                                sig = transaction_info.value.signature,
                                strategy = ?opportunity.strategy,
                                decision = ?decision,
                                "Processed opportunity with decision"
                            );
                        }
                    }
                },
                Ok(_) => {
                    trace!(target: "listen_bot", sig = transaction_info.value.signature, "No opportunities found");
                },
                Err(e) => {
                    warn!(target: "listen_bot", error = %e, sig = transaction_info.value.signature, "Error evaluating transaction");
                }
            }
        }
        
        warn!(target: "listen_bot", "Transaction log stream ended unexpectedly");
        Ok(())
    }

    /// Get transaction details using the RPC client
    async fn get_transaction_details(&self, rpc_client: &Arc<RpcClient>, signature: &Signature) -> Result<Option<serde_json::Value>> {
        // Get transaction details
        match rpc_client.get_transaction(signature, UiTransactionEncoding::Json) {
            Ok(tx) => {
                let tx_value = serde_json::to_value(tx)?;
                Ok(Some(tx_value))
            },
            Err(e) => {
                // This is often expected for very recent transactions
                debug!(target: "listen_bot", error = %e, signature = %signature, "Transaction not found via RPC");
                Ok(None)
            }
        }
    }
    
    /// Check if transaction interacts with a known DEX
    fn is_dex_interaction(&self, tx_data: &serde_json::Value) -> bool {
        // Look for DEX program IDs in transaction
        if let Some(logs) = tx_data
            .get("transaction")
            .and_then(|tx| tx.get("meta"))
            .and_then(|meta| meta.get("logMessages"))
            .and_then(|logs| logs.as_array())
        {
            // Extract program IDs from logs
            for log in logs {
                if let Some(log_str) = log.as_str() {
                    // Check for known DEX programs
                    if log_str.contains("Program JUP") ||  // Jupiter
                       log_str.contains("Program 9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin") || // Serum
                       log_str.contains("Program srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX") || // OpenBook
                       log_str.contains("Program 675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8") || // Raydium
                       log_str.contains("Program whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc")   // Orca
                    {
                        return true;
                    }
                }
            }
        }
        
        false
    }
    
    /// Process a detected MEV opportunity
    /// This is now a wrapper around OpportunityEvaluator's process_mev_opportunity
    async fn process_opportunity(&self, mut opportunity: MevOpportunity) -> Result<()> {
        // Log the opportunity
        info!(
            target: "listen_bot",
            strategy = ?opportunity.strategy,
            profit = opportunity.estimated_profit,
            "Processing MEV opportunity"
        );
        
        // Get evaluator from self
        if let Some(evaluator) = &self.evaluator {
            // Let the OpportunityEvaluator process the opportunity
            match evaluator.process_mev_opportunity(&mut opportunity).await {
                Ok(Some(signature)) => {
                    info!(
                        target: "listen_bot",
                        signature = %signature,
                        "Successfully executed MEV opportunity"
                    );
                },
                Ok(None) => {
                    info!(
                        target: "listen_bot", 
                        decision = ?opportunity.decision,
                        "Decided not to execute opportunity"
                    );
                },
                Err(e) => {
                    warn!(
                        target: "listen_bot",
                        error = %e,
                        "Error processing opportunity"
                    );
                }
            }
        } else {
            warn!(target: "listen_bot", "No evaluator available to process opportunity");
        }
        
        Ok(())
    }

    /// Set whether to use mempool or confirmed blocks
    pub fn set_use_mempool(&mut self, use_mempool: bool) {
        info!("Setting ListenBot to use mempool: {}", use_mempool);
        self.use_mempool = use_mempool;
    }
    
    /// Get whether mempool is being used
    pub fn is_using_mempool(&self) -> bool {
        self.use_mempool
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