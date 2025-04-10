use crate::listen_bot::transaction::{DexTransaction, DexTransactionParser, TransactionMonitor, TransactionInfo, TokenAmount, TransactionEvent};
use crate::listen_bot::dex::DexType;
use anyhow::Result;
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
    async fn process_transaction(
        &mut self, 
        tx_info: TransactionInfo
    ) -> crate::error::Result<()> {
        // Convert to TransactionEvent for filtering
        let event = match TransactionEvent::try_from(tx_info.clone()) {
            Ok(event) => event,
            Err(e) => return Err(e),
        };
        
        // Manual filter check based on program_id
        let filter_check = {
            let _program_id_str = event.program_id.to_string();
            // Get config from TransactionMonitor and check program_ids
            // As a fallback, just process all transactions
            true 
        };
        
        if !filter_check {
            return Ok(());
        }

        for parser in &self.parsers {
            if let Some(event) = parser.parse_transaction(tx_info.clone()).await? {
                // Convert TransactionEvent to DexTransaction
                let dex_tx = DexTransaction {
                    signature: event.signature.clone(),
                    program_id: event.program_id.clone(),
                    input_token: TokenAmount {
                        mint: event.token_mint.map_or("".to_string(), |m| m.to_string()),
                        amount: event.amount.unwrap_or(0),
                        decimals: 6, // Default
                    },
                    output_token: TokenAmount {
                        mint: "".to_string(),
                        amount: 0,
                        decimals: 6,
                    },
                    dex_name: parser.dex_name().to_string(),
                    timestamp: 0,
                    succeeded: event.success,
                    fee: 0,
                    slippage: 0.0,
                    dex_metadata: std::collections::HashMap::new(),
                    dex_type: DexType::Jupiter,
                };

                if let Err(e) = self.tx_sender.send(dex_tx) {
                    return Err(crate::error::SandoError::InternalError(format!("Failed to send transaction: {}", e)));
                }
            }
        }

        Ok(())
    }
} 