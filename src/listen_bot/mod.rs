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
    UiMessage,  // Add this import for pattern matching on message types
    UiParsedMessage,  // Add this for accessing the parsed message fields
    UiInstruction,    // Add this for working with instruction types
    UiParsedInstruction, // Add UiParsedInstruction for proper pattern matching
};
use solana_transaction_status::parse_accounts::ParsedAccount;  // Import from correct module
use solana_sdk::transaction::VersionedTransaction;
use tokio::task; // Import for spawn_blocking
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _}; // Add Engine trait
use std::time::Duration; // Add this for retry delay
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashSet;
use serde_json::Value; // Add Value for JSON handling

mod config;
mod types;
mod dex;
mod transaction;
mod stream;

pub use config::ListenBotConfig;
pub use transaction::TransactionEvent;

/// Block filtering to reduce unnecessary processing
#[derive(Debug, Clone)]
pub struct BlockFilter {
    /// Minimum number of transactions to consider processing a block
    pub min_transactions: usize,
    /// Program IDs to monitor
    pub monitored_program_ids: HashSet<String>,
    /// Token mints to monitor
    pub monitored_token_mints: HashSet<String>,
    /// DEXes to monitor
    pub monitored_dexes: HashSet<String>,
    /// Minimum transaction value in lamports
    pub min_transaction_value: u64,
}

impl Default for BlockFilter {
    fn default() -> Self {
        // Default Jupiter program ID
        let jupiter_v4 = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4".to_string();
        let jupiter_v6 = "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB".to_string();
        
        // Default Orca Whirlpools program ID
        let orca_whirlpools = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc".to_string();
        
        // Default Raydium program IDs
        let raydium_swap = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8".to_string();
        let raydium_clmm = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK".to_string();
        
        // Common tokens
        let usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string();
        let sol = "So11111111111111111111111111111111111111112".to_string();
        let msol = "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So".to_string();
        let bonk = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263".to_string();
        
        let mut program_ids = HashSet::new();
        program_ids.insert(jupiter_v4);
        program_ids.insert(jupiter_v6);
        program_ids.insert(orca_whirlpools);
        program_ids.insert(raydium_swap);
        program_ids.insert(raydium_clmm);
        
        let mut token_mints = HashSet::new();
        token_mints.insert(usdc);
        token_mints.insert(sol);
        token_mints.insert(msol);
        token_mints.insert(bonk);
        
        let mut dexes = HashSet::new();
        dexes.insert("Jupiter".to_string());
        dexes.insert("Orca".to_string());
        dexes.insert("Raydium".to_string());
        
        Self {
            min_transactions: 1,
            monitored_program_ids: program_ids,
            monitored_token_mints: token_mints,
            monitored_dexes: dexes,
            min_transaction_value: 1_000_000, // 1 SOL in lamports
        }
    }
}

impl BlockFilter {
    /// Creates a new BlockFilter with custom configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a program ID to monitor
    pub fn add_program_id(&mut self, program_id: impl Into<String>) {
        self.monitored_program_ids.insert(program_id.into());
    }
    
    /// Add a token mint to monitor
    pub fn add_token_mint(&mut self, token_mint: impl Into<String>) {
        self.monitored_token_mints.insert(token_mint.into());
    }
    
    /// Add a DEX to monitor
    pub fn add_dex(&mut self, dex: impl Into<String>) {
        self.monitored_dexes.insert(dex.into());
    }
    
    /// Set minimum number of transactions to consider processing a block
    pub fn with_min_transactions(mut self, min_transactions: usize) -> Self {
        self.min_transactions = min_transactions;
        self
    }
    
    /// Set minimum transaction value in lamports
    pub fn with_min_transaction_value(mut self, min_value: u64) -> Self {
        self.min_transaction_value = min_value;
        self
    }
    
    /// Check if a block should be processed based on preliminary transaction count
    pub fn should_process_block(&self, transaction_count: usize) -> bool {
        transaction_count >= self.min_transactions
    }
    
    /// Check if a transaction should be processed based on its program IDs and accounts
    pub fn should_process_transaction(&self, program_ids: &[String], account_keys: &[String]) -> bool {
        // Skip if no program IDs match our monitored list
        let has_monitored_program = program_ids.iter()
            .any(|id| self.monitored_program_ids.contains(id));
            
        if !has_monitored_program {
            return false;
        }
        
        // Skip if no account keys match our monitored token mints
        let has_monitored_token = account_keys.iter()
            .any(|key| self.monitored_token_mints.contains(key));
            
        if !has_monitored_token {
            return false;
        }
        
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
                encoding: Some(UiTransactionEncoding::Base64),
                transaction_details: Some(solana_transaction_status::TransactionDetails::Signatures),
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
                    maybe_slot = slot_stream.next() => {
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
                                        if let Some(transactions) = block.transactions {
                                            let tx_count = transactions.len();
                                            transactions_seen += tx_count as u64;
                                            
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
                                                
                                                // Get transaction account keys for filtering
                                                let account_keys: Vec<String> = match &tx_with_meta.transaction {
                                                    EncodedTransaction::Json(json_tx) => {
                                                        // Access account_keys through the message field by pattern matching
                                                        match &json_tx.message {
                                                            UiMessage::Parsed(parsed_msg) => {
                                                                // Convert ParsedAccount to String using pubkey field
                                                                parsed_msg.account_keys.iter()
                                                                    .map(|account| account.pubkey.clone())
                                                                    .collect()
                                                            },
                                                            UiMessage::Raw(_) => Vec::new(),
                                                        }
                                                    },
                                                    _ => Vec::new(),
                                                };
                                                
                                                // Get program IDs for filtering (from instructions)
                                                let program_ids: Vec<String> = match &tx_with_meta.transaction {
                                                    EncodedTransaction::Json(json_tx) => {
                                                        // Extract program IDs from instructions by pattern matching on message first
                                                        match &json_tx.message {
                                                            UiMessage::Parsed(parsed_msg) => {
                                                                parsed_msg.instructions.iter()
                                                                    .filter_map(|ix| {
                                                                        match ix {
                                                                            UiInstruction::Compiled(compiled_ix) => {
                                                                                // program_id_index is a u8, not an Option
                                                                                let program_idx = compiled_ix.program_id_index;
                                                                                if let UiMessage::Parsed(ref parsed_msg) = json_tx.message {
                                                                                    if program_idx < parsed_msg.account_keys.len() as u8 {
                                                                                        let program_id = parsed_msg.account_keys[program_idx as usize].pubkey.clone();
                                                                                        Some(program_id)
                                                                                    } else {
                                                                                        None
                                                                                    }
                                                                                } else {
                                                                                    None
                                                                                }
                                                                            },
                                                                            UiInstruction::Parsed(parsed_ix) => {
                                                                                // Extract program_id based on the variant
                                                                                match parsed_ix {
                                                                                    UiParsedInstruction::Parsed(parsed) => {
                                                                                        // For fully parsed instructions, program_id is in the program field
                                                                                        Some(parsed.program_id.clone())
                                                                                    },
                                                                                    UiParsedInstruction::PartiallyDecoded(partial) => {
                                                                                        // For partially decoded instructions, program_id is available directly
                                                                                        Some(partial.program_id.to_string())
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    })
                                                                    .collect()
                                                            },
                                                            UiMessage::Raw(_) => Vec::new(),
                                                        }
                                                    },
                                                    _ => Vec::new(),
                                                };
                                                
                                                // Apply transaction filtering
                                                if !block_filter.should_process_transaction(&program_ids, &account_keys) {
                                                    continue;
                                                }
                                                
                                                transactions_processed += 1;
                                                
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
                                                    
                                                    // Calculate transaction value if possible
                                                    let transaction_value = tx_with_meta.meta.as_ref()
                                                        .and_then(|meta_data| meta_data.pre_balances.get(0))
                                                        .and_then(|pre| {
                                                            tx_with_meta.meta.as_ref()
                                                                .and_then(|meta_data| meta_data.post_balances.get(0))
                                                                .map(|post| pre.saturating_sub(*post))
                                                        })
                                                        .unwrap_or(0);
                                                    
                                                    // Filter by transaction value
                                                    if transaction_value < block_filter.min_transaction_value {
                                                        debug!(tx_idx = tx_idx, %signature, value = transaction_value, "Skipping low-value transaction");
                                                        continue;
                                                    }
                                                    
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
                                                            "value": transaction_value,
                                                            "program_ids": program_ids,
                                                            "account_keys": account_keys,
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
            self.block_filter.monitored_program_ids = program_ids.into_iter().collect();
            updated = true;
        }
        
        if let Some(token_mints) = token_mints {
            info!("Updating monitored token mints, new count: {}", token_mints.len());
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
            encoding: Some(UiTransactionEncoding::Base64),
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
                            if let Some(meta) = &tx.meta {
                                if let (Some(pre), Some(post)) = (meta.pre_balances.get(0), meta.post_balances.get(0)) {
                                    let value = pre.saturating_sub(*post);
                                    if value > 0 {
                                        transaction_sizes.push(value);
                                    }
                                }
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