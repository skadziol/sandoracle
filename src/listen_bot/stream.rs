use crate::listen_bot::transaction::{DexTransaction, DexTransactionParser, TransactionMonitor, TransactionInfo, TokenAmount, TransactionEvent};
use crate::listen_bot::dex::DexType;
use anyhow::Result;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_client::nonblocking::pubsub_client::PubsubClient;
use solana_sdk::commitment_config::CommitmentConfig;
use std::{sync::Arc, time::Duration};
use tokio::sync::broadcast;
use tracing::{error, info, warn, debug};
use tokio_tungstenite::connect_async;
use futures_util::StreamExt;
use serde_json::json;
use solana_sdk::signature::Signature;
use std::str::FromStr;
use tokio::sync::mpsc::{self, Sender};
use solana_transaction_status::{
    EncodedTransaction, 
    UiTransactionEncoding, 
    TransactionDetails,
    UiConfirmedBlock 
};
use solana_client::rpc_config::RpcBlockConfig;
use solana_sdk::transaction::VersionedTransaction;
use tokio::task;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use bs58;
use tracing::trace;
use crate::error::SandoError;

/// Configuration for the transaction stream
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// RPC endpoint URL
    pub rpc_url: String,
    /// WebSocket endpoint URL
    pub ws_url: String,
    /// Commitment level for transactions
    pub commitment: CommitmentConfig,
    /// Maximum number of concurrent requests
    pub max_concurrent_requests: usize,
    /// Reconnection delay on failure
    pub reconnect_delay: Duration,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            rpc_url: "https://api.mainnet-beta.solana.com".to_string(),
            ws_url: "wss://api.mainnet-beta.solana.com".to_string(),
            commitment: CommitmentConfig::confirmed(),
            max_concurrent_requests: 10,
            reconnect_delay: Duration::from_secs(1),
        }
    }
}

/// Handles streaming and processing of transactions
pub struct TransactionStream {
    config: StreamConfig,
    monitor: TransactionMonitor,
    parsers: Vec<Arc<dyn DexTransactionParser>>,
    tx_sender: mpsc::Sender<DexTransaction>,
}

impl TransactionStream {
    /// Create a new transaction stream
    pub fn new(
        config: StreamConfig,
        monitor: TransactionMonitor,
        parsers: Vec<Arc<dyn DexTransactionParser>>,
    ) -> (Self, mpsc::Receiver<DexTransaction>) {
        let (tx_sender, rx_receiver) = mpsc::channel(1000);
        (
            Self {
                config,
                monitor,
                parsers,
                tx_sender,
            },
            rx_receiver,
        )
    }

    /// Start processing the transaction stream
    pub async fn start(self) -> Result<()> {
        info!("Starting transaction stream...");
        
        // Use PubSub client for slot subscriptions
        let pubsub_client = PubsubClient::new(&self.config.ws_url).await?;
        let (mut slot_subscription, _unsubscribe) = pubsub_client.slot_subscribe().await?;

        // Keep RPC client for fetching blocks
        let rpc_client = Arc::new(RpcClient::new_with_commitment(
            self.config.rpc_url.clone(),
            CommitmentConfig::confirmed(),
        ));

        info!("TransactionStream started, subscribing to slots...");

        while let Some(slot_info) = slot_subscription.next().await {
            let slot = slot_info.slot;
            debug!(slot, "Processing slot");

            let rpc_client_clone = rpc_client.clone();
            let parsers_clone = self.parsers.clone();
            let tx_sender_clone = self.tx_sender.clone();
            
            tokio::spawn(async move {
                let block_config = RpcBlockConfig {
                    encoding: Some(UiTransactionEncoding::Base64),
                    transaction_details: Some(TransactionDetails::Full),
                    rewards: Some(false),
                    commitment: Some(CommitmentConfig::confirmed()),
                    max_supported_transaction_version: Some(0),
                };

                match Self::fetch_block_with_retry(rpc_client_clone.clone(), slot, block_config, 3, 100).await {
                    Ok(Some(block)) => {
                        if let Some(transactions) = block.transactions {
                            for tx_with_meta in transactions {
                                // Store transaction in a variable to avoid multiple clones
                                let transaction = tx_with_meta.transaction.clone();
                                
                                if let (EncodedTransaction::LegacyBinary(tx_b58), Some(meta)) = 
                                    (transaction, tx_with_meta.meta.clone())
                                {
                                    match bs58::decode(tx_b58).into_vec() {
                                        Ok(tx_bytes_vec) => {
                                            match bincode::deserialize::<solana_sdk::transaction::Transaction>(&tx_bytes_vec) {
                                                Ok(decoded_tx) => {
                                                    let tx_info = TransactionInfo {
                                                        transaction: VersionedTransaction::from(decoded_tx),
                                                        meta: Some(meta),
                                                        signature: "legacy_tx_sig_placeholder".to_string(),
                                                        program_id: "placeholder_program".to_string(),
                                                        token_mint: None,
                                                        amount: None,
                                                        success: tx_with_meta.meta.map_or(false, |m| m.status.is_ok()),
                                                    };
                                                    
                                                    for parser in &parsers_clone {
                                                        match parser.parse_transaction(tx_info.clone()).await {
                                                            Ok(Some(dex_tx)) => {
                                                                if let Err(e) = tx_sender_clone.send(dex_tx).await {
                                                                    error!(error = %e, "Failed to send parsed DexTransaction to channel");
                                                                }
                                                                break;
                                                            }
                                                            Ok(None) => {}
                                                            Err(e) => {
                                                                error!(error = %e, parser = parser.dex_name(), "Parser failed for transaction");
                                                            }
                                                        }
                                                    }
                                                }
                                                Err(e) => error!(error = %e, "Failed to deserialize legacy transaction bytes"),
                                            }
                                        }
                                         Err(e) => error!(error = %e, "Failed to decode base58 legacy transaction string"),
                                    }
                                } else if let (EncodedTransaction::Binary(tx_b64, _encoding), Some(meta)) = 
                                    (tx_with_meta.transaction.clone(), tx_with_meta.meta.clone())
                                {
                                    match BASE64_STANDARD.decode(&tx_b64) {
                                        Ok(tx_bytes_vec) => {
                                             match bincode::deserialize::<VersionedTransaction>(&tx_bytes_vec) {
                                                Ok(versioned_tx) => {
                                                    let signature = versioned_tx.signatures.get(0).cloned().unwrap_or_default();
                                                      let tx_info = TransactionInfo {
                                                        transaction: versioned_tx,
                                                        meta: Some(meta),
                                                        signature: signature.to_string(),
                                                        program_id: "placeholder_program".to_string(),
                                                        token_mint: None,
                                                        amount: None,
                                                        success: tx_with_meta.meta.map_or(false, |m| m.status.is_ok()),
                                                    };
                                                    for parser in &parsers_clone {
                                                         match parser.parse_transaction(tx_info.clone()).await {
                                                            Ok(Some(dex_tx)) => {
                                                                 if let Err(e) = tx_sender_clone.send(dex_tx).await {
                                                                    error!(error = %e, "Failed to send parsed DexTransaction to channel");
                                                                }
                                                                break;
                                                            }
                                                            Ok(None) => {}
                                                            Err(e) => {
                                                                error!(error = %e, parser = parser.dex_name(), "Parser failed for transaction");
                                                            }
                                                        }
                                                    }
                                                }
                                                 Err(e) => error!(error = %e, "Failed to deserialize versioned transaction bytes"),
                                            }
                                        }
                                         Err(e) => error!(error = %e, "Failed to decode base64 versioned transaction string"),
                                    }
                                } else {
                                    trace!("Skipping transaction with unhandled encoding or missing meta");
                                }
                            }
                        }
                    }
                    Ok(None) => { /* Block not found or fetch timed out */ }
                    Err(e) => error!(slot, error = %e, "Failed to fetch or process block"),
                }
            });
        }

        info!("TransactionStream stopped.");
        Ok(())
    }

    async fn fetch_block_with_retry(
        rpc_client: Arc<RpcClient>,
        slot: u64,
        config: RpcBlockConfig,
        max_retries: u32,
        initial_delay_ms: u64,
    ) -> Result<Option<UiConfirmedBlock>, SandoError> {
        let mut retries = 0;
        let mut delay_ms = initial_delay_ms;
        
        loop {
            let config_clone = config.clone();
            let client = rpc_client.clone();
            
            // Await the JoinHandle first
            let blocking_result = task::spawn_blocking(move || {
                // Use the blocking version to avoid returning a Future
                let result = futures::executor::block_on(client.get_block_with_config(slot, config_clone));
                result
            }).await;

            // Match on the Result from spawn_blocking (handles JoinError)
            match blocking_result {
                Ok(rpc_result) => { 
                    // Now match on the Result from get_block_with_config
                    match rpc_result {
                        Ok(block) => return Ok(Some(block)),
                        Err(err) => { // ClientError
                            if retries >= max_retries {
                                error!(slot, error = %err, max_retries, "Failed to fetch block after max retries");
                                return Err(SandoError::SolanaRpc(format!("Failed to fetch block {} after {} retries: {}", slot, max_retries, err)));
                            }
                            
                            let err_str = err.to_string();
                            let is_block_not_available = err_str.contains("Block not available") || err_str.contains("-32004");
                            
                            if !is_block_not_available {
                                error!(slot, error = %err, "Non-retriable error fetching block");
                                return Err(SandoError::SolanaRpc(format!("Non-retriable error fetching block: {}", err)));
                            }
                            
                            warn!(slot, retry_count = retries + 1, max_retries, delay_ms, "Block not available, retrying...");
                            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                            retries += 1;
                            delay_ms = std::cmp::min(delay_ms * 2, 2000); 
                        }
                    }
                }
                Err(e) => { // JoinError (task panicked)
                    error!(slot, error = %e, "Block fetching task panicked");
                    return Err(SandoError::InternalError(format!("Block fetching task panicked: {}", e)));
                }
            }
        }
    }
} 