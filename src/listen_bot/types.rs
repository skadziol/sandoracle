use solana_sdk::{signature::Signature, pubkey::Pubkey};
use solana_transaction_status::{UiTransactionStatusMeta, EncodedConfirmedTransactionWithStatusMeta};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Represents the current state of the connection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Initial state
    Initial,
    /// Attempting to connect
    Connecting,
    /// Successfully connected
    Connected,
    /// Connection lost, attempting to reconnect
    Reconnecting,
    /// Connection closed
    Closed,
}

/// Type of transaction being processed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionType {
    /// Token swap transaction
    Swap,
    /// Arbitrage transaction
    Arbitrage,
    /// Snipe transaction
    Snipe,
    /// Unknown transaction type
    Unknown,
}

/// Represents a processed transaction event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionEvent {
    /// Transaction signature
    pub signature: String,
    /// Transaction status and metadata
    pub status: UiTransactionStatusMeta,
    /// Transaction type
    pub tx_type: TransactionType,
    /// Program IDs involved in the transaction
    pub program_ids: Vec<Pubkey>,
    /// Token mints involved in the transaction
    pub token_mints: Vec<Pubkey>,
    /// Transaction amount in lamports
    pub amount: u64,
    /// Additional transaction metadata
    pub metadata: TransactionMetadata,
    /// Timestamp when the event was received
    pub timestamp: i64,
    /// Whether the transaction was successful
    pub success: bool,
    /// Transaction fee in lamports
    pub fee: u64,
}

/// Additional metadata about the transaction
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransactionMetadata {
    /// Additional transaction-specific data
    pub extra: HashMap<String, String>,
}

/// Statistics about the WebSocket connection
#[derive(Debug, Default, Clone)]
pub struct ConnectionStats {
    /// Time when the connection was established
    pub connected_at: Option<Instant>,
    /// Number of successful transactions processed
    pub transactions_processed: u64,
    /// Number of failed transactions
    pub transactions_failed: u64,
    /// Average transaction processing time in milliseconds
    pub avg_processing_time: f64,
    /// Number of reconnection attempts
    pub reconnection_attempts: u32,
}

impl ConnectionStats {
    /// Records a successful connection
    pub fn record_connection(&mut self) {
        self.connected_at = Some(Instant::now());
    }

    /// Records a processed transaction with its processing time
    pub fn record_transaction(&mut self, processing_time_ms: f64, failed: bool) {
        if failed {
            self.transactions_failed += 1;
        } else {
            self.transactions_processed += 1;
        }

        // Update average processing time using moving average
        let total = self.transactions_processed + self.transactions_failed;
        self.avg_processing_time = (self.avg_processing_time * (total - 1) as f64 + processing_time_ms) / total as f64;
    }

    /// Records a reconnection attempt
    pub fn record_reconnection(&mut self) {
        self.reconnection_attempts += 1;
    }

    /// Returns the uptime of the connection
    pub fn uptime(&self) -> Option<Duration> {
        self.connected_at.map(|t| t.elapsed())
    }
}

impl TransactionEvent {
    /// Creates a new TransactionEvent from a confirmed transaction
    pub fn from_confirmed_tx(
        tx: EncodedConfirmedTransactionWithStatusMeta,
        timestamp: i64,
    ) -> Option<Self> {
        let signature = tx.transaction.transaction.signatures[0].to_string();
        let meta = tx.transaction.meta?;
        
        let success = meta.err.is_none();
        let fee = meta.fee;
        
        // Extract program IDs and token mints from the transaction
        let program_ids = get_program_accounts(&tx);
        let token_mints = get_token_accounts(&tx);

        Some(Self {
            signature,
            status: meta,
            tx_type: TransactionType::Unknown,
            program_ids,
            token_mints,
            amount: 0, // Will be set by specific handlers
            metadata: TransactionMetadata {
                extra: HashMap::new(),
            },
            timestamp,
            success,
            fee,
        })
    }

    /// Returns whether this transaction was successful
    pub fn is_successful(&self) -> bool {
        self.success
    }

    /// Returns the transaction fee in lamports
    pub fn fee(&self) -> u64 {
        self.fee
    }

    /// Returns whether this transaction involves a specific program
    pub fn involves_program(&self, program_id: &Pubkey) -> bool {
        self.program_ids.contains(program_id)
    }

    /// Returns whether this transaction involves a specific token
    pub fn involves_token(&self, mint: &Pubkey) -> bool {
        self.token_mints.contains(mint)
    }
}

fn get_program_accounts(tx: &EncodedConfirmedTransactionWithStatusMeta) -> Vec<Pubkey> {
    tx.transaction.transaction.message.account_keys.iter()
        .filter(|key| tx.transaction.meta.as_ref().map_or(false, |meta| {
            meta.post_token_balances.iter().any(|balance| balance.owner == key.to_string())
        }))
        .cloned()
        .collect()
}

fn get_token_accounts(tx: &EncodedConfirmedTransactionWithStatusMeta) -> Vec<Pubkey> {
    tx.transaction.transaction.message.account_keys.iter()
        .filter(|key| {
            tx.transaction.meta.as_ref().map_or(false, |meta| {
                meta.post_token_balances.iter().any(|balance| balance.account_index == key.to_string())
            })
        })
        .cloned()
        .collect()
} 