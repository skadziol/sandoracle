use crate::listen_bot::transaction::{DexTransaction, DexTransactionParser, TransactionMonitor, TransactionInfo, TokenAmount, TransactionEvent};
use crate::listen_bot::dex::DexType;
use anyhow::Result;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::commitment_config::CommitmentConfig;
use std::{sync::Arc, time::Duration};
use tokio::sync::broadcast;
use tracing::{error, info, warn, debug};
use tokio_tungstenite::connect_async;
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use solana_sdk::signature::Signature;
use std::str::FromStr;

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
        // DEX program IDs (mainnet)
        // These are examples; replace with actual program IDs as needed
        const ORCA_PROGRAM_ID: &str = "9WwGQq5FQn6rQp6QwBEmcLk6RdtqvYXvyfZ7u5nE5fSL";
        const RAYDIUM_PROGRAM_ID: &str = "4k3Dyjzvzp8e2A6A6bJ4hF6dkprdFM5ocTyT4p7gX9E5";
        const JUPITER_PROGRAM_ID: &str = "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB";

        let ws_url = &self.config.ws_url;
        info!("Connecting to Solana WebSocket endpoint: {}", ws_url);
        let (ws_stream, _) = connect_async(ws_url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Subscribe to logs for each DEX program
        let subscriptions = vec![
            ORCA_PROGRAM_ID,
            RAYDIUM_PROGRAM_ID,
            JUPITER_PROGRAM_ID,
        ];
        for program_id in &subscriptions {
            let sub_msg = json!([
                "logsSubscribe",
                { "mentions": [program_id] },
                { "commitment": "confirmed" }
            ]);
            let sub_text = serde_json::to_string(&sub_msg)?;
            write.send(async_tungstenite::tungstenite::Message::Text(sub_text)).await?;
            info!("Subscribed to logs for program_id: {}", program_id);
        }

        // Listen for log notifications
        while let Some(msg) = read.next().await {
            debug!("Raw WebSocket message: {:?}", msg);
            match msg {
                Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                    info!(target: "solana_ws", "Received log notification: {}", text);
                    // Parse the log notification JSON
                    let parsed: serde_json::Value = match serde_json::from_str(&text) {
                        Ok(val) => val,
                        Err(e) => {
                            warn!("Failed to parse log notification JSON: {}", e);
                            continue;
                        }
                    };
                    // Extract the signature from the notification
                    let maybe_sig = parsed
                        .get("params")
                        .and_then(|params| params.get("result"))
                        .and_then(|result| result.get("signature"))
                        .and_then(|sig| sig.as_str());
                    if let Some(signature_str) = maybe_sig {
                        // Convert signature string to Signature type
                        let signature = match Signature::from_str(signature_str) {
                            Ok(sig) => sig,
                            Err(e) => {
                                warn!("Invalid signature string: {}", e);
                                continue;
                            }
                        };
                        // Fetch the full transaction and metadata
                        match self.rpc_client.get_transaction(&signature, solana_transaction_status::UiTransactionEncoding::Json).await {
                            Ok(tx_status) => {
                                // Build TransactionInfo (simplified for now)
                                if let Some(tx) = tx_status.transaction.transaction.decode() {
                                    let meta = tx_status.transaction.meta.clone();
                                    let program_id = tx.message.static_account_keys().get(0).map(|k| k.to_string()).unwrap_or_default();
                                    let token_mint = None; // TODO: Parse from instructions
                                    let amount = None; // TODO: Parse from instructions
                                    let success = tx_status.transaction.meta.as_ref().map(|m| m.status.is_ok()).unwrap_or(false);
                                    let tx_info = TransactionInfo {
                                        transaction: tx,
                                        meta,
                                        signature: signature_str.to_string(),
                                        program_id,
                                        token_mint,
                                        amount,
                                        success,
                                    };
                                    // Call process_transaction (mutable borrow workaround)
                                    match self.process_transaction(tx_info).await {
                                        Ok(_) => info!("Processed transaction for signature: {}", signature_str),
                                        Err(e) => warn!("Failed to process transaction: {}", e),
                                    }
                                } else {
                                    warn!("Failed to decode transaction for signature: {}", signature_str);
                                }
                            }
                            Err(e) => {
                                warn!("Failed to fetch transaction for signature {}: {}", signature_str, e);
                            }
                        }
                    } else {
                        warn!("No signature found in log notification");
                    }
                }
                Ok(other) => {
                    debug!("Non-text WebSocket message: {:?}", other);
                }
                Err(e) => {
                    error!("WebSocket error: {}", e);
                    break;
                }
            }
        }
        Ok(())
    }

    /// Process a single transaction
    async fn process_transaction(
        &self, 
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