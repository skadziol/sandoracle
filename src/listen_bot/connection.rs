use crate::error::{Result, SandoError};
use super::{
    config::ListenBotConfig,
    types::{ConnectionState, ConnectionStats, TransactionEvent, TransactionType},
};
use tokio::{
    sync::{broadcast, RwLock},
    time::{sleep, Duration, Instant},
};
use solana_client::{
    nonblocking::rpc_client::RpcClient,
    rpc_config::RpcTransactionConfig,
};
use solana_client_ws::client_ws::{ClientWs, Response};
use solana_transaction_status::EncodedConfirmedTransactionWithStatusMeta;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Manages WebSocket connection to Solana RPC
pub struct ListenBotConnection {
    /// Configuration for the connection
    config: ListenBotConfig,
    /// Current connection state
    state: Arc<RwLock<ConnectionState>>,
    /// Connection statistics
    stats: Arc<RwLock<ConnectionStats>>,
    /// Broadcast channel for transaction events
    tx_sender: broadcast::Sender<TransactionEvent>,
    /// RPC client for HTTP requests
    rpc_client: RpcClient,
}

impl ListenBotConnection {
    /// Creates a new ListenBotConnection with the given configuration
    pub fn new(config: ListenBotConfig) -> Self {
        let (tx_sender, _) = broadcast::channel(super::DEFAULT_CHANNEL_CAPACITY);
        
        Self {
            rpc_client: RpcClient::new(config.rpc_url.clone()),
            config,
            state: Arc::new(RwLock::new(ConnectionState::Initial)),
            stats: Arc::new(RwLock::new(ConnectionStats::default())),
            tx_sender,
        }
    }

    /// Starts the WebSocket connection with automatic reconnection
    pub async fn start(&self) -> Result<()> {
        let mut current_retry = 0;
        let mut backoff = self.config.initial_backoff_ms;

        loop {
            *self.state.write().await = ConnectionState::Connecting;
            
            match self.connect().await {
                Ok(_) => {
                    info!("Successfully connected to WebSocket endpoint");
                    self.stats.write().await.record_connection();
                    *self.state.write().await = ConnectionState::Connected;
                    current_retry = 0;
                    backoff = self.config.initial_backoff_ms;
                },
                Err(e) => {
                    error!("Failed to connect: {}", e);
                    self.stats.write().await.record_error();
                    
                    if current_retry >= self.config.max_retries {
                        *self.state.write().await = ConnectionState::Failed;
                        return Err(SandoError::connection_failed("Max retries exceeded"));
                    }

                    *self.state.write().await = ConnectionState::Reconnecting;
                    self.stats.write().await.record_reconnection();
                    
                    warn!("Reconnecting in {}ms (attempt {}/{})", 
                          backoff, current_retry + 1, self.config.max_retries);
                    
                    sleep(Duration::from_millis(backoff)).await;
                    
                    current_retry += 1;
                    backoff = std::cmp::min(
                        backoff * 2,
                        self.config.max_backoff_ms
                    );
                }
            }
        }
    }

    /// Establishes a WebSocket connection and sets up message handling
    async fn connect(&self) -> Result<()> {
        let ws_client = ClientWs::new(&self.config.ws_url)
            .await
            .map_err(|e| SandoError::connection_failed(e.to_string()))?;

        let subscribe_config = RpcTransactionConfig {
            commitment: Some(self.config.commitment),
            max_supported_transaction_version: Some(0),
            encoding: None,
        };

        ws_client.transaction_subscribe(subscribe_config, |response| {
            let start_time = Instant::now();
            self.stats.write().await.record_message();

            match self.process_transaction(response).await {
                Ok(Some(event)) => {
                    let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
                    
                    // Apply filters
                    if self.should_process_transaction(&event) {
                        if let Err(e) = self.tx_sender.send(event) {
                            error!("Failed to broadcast transaction event: {}", e);
                        }
                        self.stats.write().await.record_transaction(processing_time, false);
                    } else {
                        debug!("Transaction filtered out: {}", event.signature);
                        self.stats.write().await.record_transaction(processing_time, true);
                    }
                }
                Ok(None) => {
                    debug!("Transaction skipped (invalid or incomplete data)");
                }
                Err(e) => {
                    error!("Failed to process transaction: {}", e);
                    self.stats.write().await.record_error();
                }
            }
        })
        .await
        .map_err(|e| SandoError::connection_failed(e.to_string()))?;

        Ok(())
    }

    /// Processes an incoming transaction and converts it to a TransactionEvent
    async fn process_transaction(&self, response: Response) -> Result<Option<TransactionEvent>> {
        let tx = match response {
            Response::Transaction(tx) => tx,
            _ => return Ok(None),
        };

        let timestamp = chrono::Utc::now().timestamp();
        let mut event = TransactionEvent::from_confirmed_tx(tx.clone(), timestamp)
            .ok_or_else(|| SandoError::transaction_processing("Invalid transaction data"))?;

        // Extract program IDs and token mints
        if let Some(meta) = tx.transaction_status_meta {
            event.metadata.fee = meta.fee;
            event.metadata.compute_units_consumed = meta.compute_units_consumed;
            
            // Extract program IDs from inner instructions
            if let Some(inner_instructions) = meta.inner_instructions {
                for instruction in inner_instructions {
                    if let Some(program_id) = instruction.instruction.program_id {
                        event.program_ids.push(program_id);
                    }
                }
            }

            // Extract token mints from token balances
            if let Some(token_balances) = meta.pre_token_balances {
                for balance in token_balances {
                    if let Some(mint) = balance.mint {
                        event.token_mints.push(mint);
                    }
                }
            }

            // Determine transaction type based on program IDs and instructions
            event.tx_type = self.determine_transaction_type(&event);
        }

        Ok(Some(event))
    }

    /// Determines the type of transaction based on its contents
    fn determine_transaction_type(&self, event: &TransactionEvent) -> TransactionType {
        // This is a simplified implementation - in practice, you would:
        // 1. Check program IDs against known DEX programs
        // 2. Analyze instruction data
        // 3. Look at token balance changes
        // For now, we'll just return Unknown
        TransactionType::Unknown
    }

    /// Checks if a transaction should be processed based on filter configuration
    fn should_process_transaction(&self, event: &TransactionEvent) -> bool {
        // Skip failed transactions if not explicitly included
        if !event.is_successful() && !self.config.filter.include_failed {
            return false;
        }

        // Check transaction size constraints
        if let Some(min_size) = self.config.filter.min_size {
            if event.amount < min_size {
                return false;
            }
        }
        if let Some(max_size) = self.config.filter.max_size {
            if event.amount > max_size {
                return false;
            }
        }

        // Check program ID filters
        if !self.config.filter.program_ids.is_empty() {
            if !event.program_ids.iter().any(|id| self.config.filter.program_ids.contains(id)) {
                return false;
            }
        }

        // Check token mint filters
        if !self.config.filter.token_mints.is_empty() {
            if !event.token_mints.iter().any(|mint| self.config.filter.token_mints.contains(mint)) {
                return false;
            }
        }

        true
    }

    /// Returns a receiver for transaction events
    pub fn subscribe(&self) -> broadcast::Receiver<TransactionEvent> {
        self.tx_sender.subscribe()
    }

    /// Returns the current connection state
    pub async fn state(&self) -> ConnectionState {
        *self.state.read().await
    }

    /// Returns the current connection statistics
    pub async fn stats(&self) -> ConnectionStats {
        self.stats.read().await.clone()
    }

    /// Gracefully closes the connection
    pub async fn shutdown(&self) -> Result<()> {
        *self.state.write().await = ConnectionState::Closed;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_connection_lifecycle() {
        let config = ListenBotConfig::default();
        let connection = ListenBotConnection::new(config);

        assert_eq!(connection.state().await, ConnectionState::Initial);

        // Start connection in background
        let connection_handle = tokio::spawn({
            let connection = connection.clone();
            async move {
                connection.start().await
            }
        });

        // Wait for connection or timeout
        match timeout(Duration::from_secs(5), connection_handle).await {
            Ok(result) => {
                match result {
                    Ok(_) => panic!("Connection should not complete"),
                    Err(e) => println!("Connection failed as expected: {}", e),
                }
            }
            Err(_) => println!("Connection timeout as expected"),
        }

        // Verify stats were recorded
        let stats = connection.stats().await;
        assert!(stats.connections > 0 || stats.errors > 0);
    }

    #[tokio::test]
    async fn test_transaction_filtering() {
        let mut config = ListenBotConfig::default();
        config.filter = TransactionFilterConfig::default()
            .with_min_size(1000)
            .include_failed(false);

        let connection = ListenBotConnection::new(config);

        // Create a test transaction event
        let event = TransactionEvent {
            signature: Signature::default(),
            status: UiTransactionStatusMeta::default(),
            tx_type: TransactionType::Unknown,
            program_ids: vec![],
            token_mints: vec![],
            amount: 500, // Below min_size
            metadata: TransactionMetadata::default(),
            timestamp: 0,
        };

        // Test filtering
        assert!(!connection.should_process_transaction(&event));
    }
} 