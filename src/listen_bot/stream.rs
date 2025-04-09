use crate::listen_bot::transaction::{DexTransaction, DexTransactionParser, TransactionMonitor};
use anyhow::Result;
use futures_util::stream::StreamExt;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::commitment_config::CommitmentConfig;
use std::{sync::Arc, time::Duration};
use tokio::sync::broadcast;
use tracing::{error, info, warn};

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
    monitor: Arc<TransactionMonitor>,
    parsers: Vec<Arc<dyn DexTransactionParser>>,
    rpc_client: Arc<RpcClient>,
    tx_sender: broadcast::Sender<DexTransaction>,
}

impl TransactionStream {
    /// Create a new transaction stream
    pub fn new(
        config: StreamConfig,
        monitor: TransactionMonitor,
        parsers: Vec<Arc<dyn DexTransactionParser>>,
    ) -> (Self, broadcast::Receiver<DexTransaction>) {
        let (tx_sender, tx_receiver) = broadcast::channel(1000);
        let rpc_client = Arc::new(RpcClient::new(config.rpc_url.clone()));

        (
            Self {
                config,
                monitor: Arc::new(monitor),
                parsers,
                rpc_client,
                tx_sender,
            },
            tx_receiver,
        )
    }

    /// Start processing the transaction stream
    pub async fn start(&self) -> Result<()> {
        info!("Starting transaction stream...");
        
        loop {
            match self.process_stream().await {
                Ok(_) => {
                    warn!("Transaction stream ended, reconnecting...");
                }
                Err(e) => {
                    error!("Error processing transaction stream: {}", e);
                }
            }

            tokio::time::sleep(self.config.reconnect_delay).await;
            info!("Attempting to reconnect...");
        }
    }

    /// Process the transaction stream
    async fn process_stream(&self) -> Result<()> {
        // TODO: Implement WebSocket connection and transaction processing
        // This will involve:
        // 1. Connecting to the WebSocket endpoint
        // 2. Subscribing to transaction notifications
        // 3. Processing transactions in parallel
        // 4. Parsing DEX-specific information
        // 5. Broadcasting parsed transactions
        
        Ok(())
    }

    /// Process a single transaction
    async fn process_transaction(&self, signature: String) -> Result<()> {
        let tx = self.rpc_client
            .get_transaction_with_config(
                &signature.parse()?,
                solana_client::rpc_config::RpcTransactionConfig {
                    encoding: Some(solana_transaction_status::UiTransactionEncoding::Json),
                    commitment: Some(self.config.commitment),
                    max_supported_transaction_version: Some(0),
                },
            )
            .await?;

        if let Some(tx) = tx {
            // Check if we should monitor this transaction
            if !self.monitor.should_monitor(&tx.transaction, &tx.meta.unwrap()) {
                return Ok(());
            }

            // Try parsing with each DEX parser
            for parser in &self.parsers {
                if let Ok(Some(dex_tx)) = parser.parse_transaction(
                    tx.transaction.clone(),
                    tx.meta.unwrap(),
                ).await {
                    // Broadcast the parsed transaction
                    let _ = self.tx_sender.send(dex_tx);
                    break;
                }
            }
        }

        Ok(())
    }
} 