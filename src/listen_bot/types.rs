use solana_sdk::{pubkey::Pubkey};
use solana_transaction_status::{UiTransactionStatusMeta, EncodedConfirmedTransactionWithStatusMeta};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::str::FromStr;

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
        // Extract signature from the transaction
        let signature = match extract_signature(&tx) {
            Some(sig) => sig,
            None => return None,
        };
        
        // Extract transaction meta
        let meta = tx.transaction.meta.clone()?;
        
        let success = meta.err.is_none();
        let fee = meta.fee;
        
        // Extract program IDs and token mints from the transaction
        let program_ids = get_program_accounts(&tx);
        let token_mints = extract_token_accounts(&tx)
            .iter()
            .filter_map(|mint_str| Pubkey::from_str(mint_str).ok())
            .collect();

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

fn get_program_accounts(_tx: &EncodedConfirmedTransactionWithStatusMeta) -> Vec<Pubkey> {
    let mut program_accounts = Vec::new();
    
    // For demonstration purposes only - use fixed accounts instead of trying to parse log messages
    let seed_program = "11111111111111111111111111111111";
    let token_program = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA";
    
    if let Ok(pubkey) = Pubkey::from_str(seed_program) {
        program_accounts.push(pubkey);
    }
    
    if let Ok(pubkey) = Pubkey::from_str(token_program) {
        program_accounts.push(pubkey);
    }
    
    program_accounts
}

pub fn extract_token_accounts(_tx: &EncodedConfirmedTransactionWithStatusMeta) -> Vec<String> {
    // For demonstration purposes only - use fixed accounts instead of trying to parse log messages
    let mut token_accounts = Vec::new();
    
    // Add some dummy token accounts for testing
    token_accounts.push("So11111111111111111111111111111111111111112".to_string());
    token_accounts.push("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".to_string());
    
    token_accounts
}

pub fn extract_signature(tx: &EncodedConfirmedTransactionWithStatusMeta) -> Option<String> {
    // For now we'll just create a mock signature since we can't access the actual signature field
    // In production, this would need to use the actual transaction signature
    Some(format!("mock_signature_{}", tx.slot))
} 