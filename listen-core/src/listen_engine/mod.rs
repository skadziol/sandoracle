use anyhow::{Result, anyhow};
use futures::Stream;
use solana_client::rpc_client::RpcClient;
use solana_client::rpc_config::RpcTransactionConfig;
use solana_client::nonblocking::pubsub_client::PubsubClient;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::signature::Signature;
use futures::stream::StreamExt;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use tokio::time::sleep;
use tokio::time::Duration;
use tracing::{info, debug, warn, error};
use async_stream::stream;
use crate::{
    model::tx::Transaction,
    router::dexes::DexName,
};

#[derive(Debug, Clone)]
pub struct ListenEngineConfig {
    pub rpc_url: String,
    pub commitment: String,
    pub ws_url: Option<String>,
}

impl Default for ListenEngineConfig {
    fn default() -> Self {
        Self {
            rpc_url: "https://api.mainnet-beta.solana.com".to_string(),
            commitment: "confirmed".to_string(),
            ws_url: None,
        }
    }
}

pub struct ListenEngine {
    config: ListenEngineConfig,
    client: Arc<RpcClient>,
}

impl ListenEngine {
    pub fn new(config: ListenEngineConfig) -> Result<Self> {
        let client = Arc::new(RpcClient::new(config.rpc_url.clone()));
        Ok(Self { config, client })
    }

    /// Returns a clone of the underlying RPC client Arc.
    pub fn rpc_client(&self) -> Arc<RpcClient> {
        self.client.clone()
    }

    pub async fn stream_dex_swaps(
        &self,
        dexes: Vec<DexName>,
    ) -> Result<Pin<Box<dyn Stream<Item = u64> + Send + '_>>> {
        let ws_url = self.config.ws_url.clone()
            .ok_or_else(|| anyhow!("WebSocket URL is not configured"))?;
        let client = self.client.clone(); // Clone Arc for stream
        
        // Note: dexes parameter is currently unused in this placeholder implementation
        // It would be used later for filtering transactions fetched from the slot
        info!(?dexes, "Starting DEX swap stream for specified DEXes (currently placeholder)");

        let stream = async_stream::stream! {
            loop {
                info!(url = %ws_url, "Attempting to connect to Solana websocket...");
                match PubsubClient::new(&ws_url).await {
                    Ok(pubsub_client) => {
                        info!("Successfully connected to WebSocket. Subscribing to slots...");
                        match pubsub_client.slot_subscribe().await {
                            Ok((mut notifications, _unsubscribe)) => {
                                info!("Successfully subscribed to slot notifications");
                                while let Some(notification) = notifications.next().await {
                                    let slot = notification.slot;
                                    debug!(slot, "Received new slot notification");
                                    yield slot;
                                }
                                warn!("Slot notification stream ended. Will attempt to reconnect.");
                            }
                            Err(e) => {
                                error!(error = ?e, "Failed to subscribe to slot notifications");
                            }
                        }
                    }
                    Err(e) => {
                        error!(error = ?e, "Failed to connect to Solana websocket");
                    }
                }
                // Reconnection delay
                warn!("Attempting to reconnect after 1 second...");
                sleep(Duration::from_secs(1)).await;
            }
        };

        Ok(Box::pin(stream))
    }

    pub async fn get_transaction_details(&self, signature_str: &str) -> Result<Transaction> {
        info!("Fetching transaction details for signature: {}", signature_str);
        let signature = Signature::from_str(signature_str)?;
        
        let tx = self.client.get_transaction_with_config(
            &signature,
            RpcTransactionConfig {
                commitment: Some(CommitmentConfig::from_str(&self.config.commitment)?),
                encoding: None,
                max_supported_transaction_version: None,
            },
        )?;

        info!("Successfully fetched transaction details for signature: {}", signature_str);
        Ok(Transaction {
            signature: signature_str.to_string(),
            signer: solana_sdk::pubkey::new_rand(), // Placeholder
            dex: None,
            swap_info: None,
            block_time: tx.block_time,
            slot: Some(tx.slot),
        })
    }
}
