use crate::error::{Result, SandoError};
use crate::listen_bot::types::TransactionInfo;
use crate::listen_bot::config::ListenBotConfig;
use solana_sdk::{signature::Signature, pubkey::Pubkey};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;
use solana_sdk::{
    transaction::Transaction,
};
use solana_transaction_status::UiTransactionStatusMeta;
use std::fmt::Debug;
use crate::listen_bot::DexType;
use crate::error::TransactionErrorKind;

/// Represents a DEX transaction with relevant MEV information
#[derive(Debug, Clone)]
pub struct DexTransaction {
    /// Transaction signature
    pub signature: Signature,
    /// Program ID that processed the transaction
    pub program_id: Pubkey,
    /// Input token mint and amount
    pub input_token: TokenAmount,
    /// Output token mint and amount
    pub output_token: TokenAmount,
    /// DEX that processed the transaction
    pub dex_name: String,
    /// Transaction timestamp
    pub timestamp: i64,
    /// Whether the transaction succeeded
    pub succeeded: bool,
    /// Transaction fee in lamports
    pub fee: u64,
    /// Computed slippage as a percentage
    pub slippage: f64,
    /// Additional metadata specific to the DEX
    pub dex_metadata: HashMap<String, String>,
    /// DEX type
    pub dex_type: DexType,
}

/// Represents a token amount in a transaction
#[derive(Debug, Clone)]
pub struct TokenAmount {
    /// Token mint address
    pub mint: String,
    /// Token amount (raw)
    pub amount: u64,
    /// Token decimals
    pub decimals: u8,
}

impl TokenAmount {
    /// Convert raw amount to decimal value
    pub fn to_decimal(&self) -> f64 {
        self.amount as f64 / 10f64.powi(self.decimals as i32)
    }
}

#[derive(Debug, Clone)]
pub struct TransactionEvent {
    pub signature: Signature,
    pub program_id: Pubkey,
    pub token_mint: Option<Pubkey>,
    pub amount: Option<u64>,
    pub success: bool,
}

impl TryFrom<TransactionInfo> for TransactionEvent {
    type Error = SandoError;

    fn try_from(tx_info: TransactionInfo) -> Result<Self> {
        Ok(TransactionEvent {
            signature: tx_info.signature.parse()
                .map_err(|e| SandoError::transaction(
                    TransactionErrorKind::InvalidSignature,
                    format!("Invalid signature: {}", e)
                ))?,
            program_id: tx_info.program_id.parse()
                .map_err(|e| SandoError::transaction(
                    TransactionErrorKind::Other,
                    format!("Invalid program ID: {}", e)
                ))?,
            token_mint: tx_info.token_mint.map(|mint| mint.parse())
                .transpose()
                .map_err(|e| SandoError::transaction(
                    TransactionErrorKind::Other,
                    format!("Invalid token mint: {}", e)
                ))?,
            amount: tx_info.amount,
            success: tx_info.success,
        })
    }
}

/// Transaction monitoring configuration
#[derive(Debug)]
pub struct TransactionMonitor {
    config: Arc<ListenBotConfig>,
    dex_parsers: Arc<RwLock<HashMap<String, Box<dyn DexTransactionParser + Send + Sync>>>>,
}

impl TransactionMonitor {
    /// Create a new transaction monitor with the given configuration
    pub fn new(config: ListenBotConfig) -> Self {
        TransactionMonitor {
            config: Arc::new(config),
            dex_parsers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Check if a transaction should be monitored based on configuration
    pub async fn process_transaction(&self, tx_info: TransactionInfo) -> Result<Option<TransactionEvent>> {
        let event = TransactionEvent::try_from(tx_info)?;
        
        // Apply transaction filters
        if !self.should_process_transaction(&event).await {
            return Ok(None);
        }

        Ok(Some(event))
    }

    async fn should_process_transaction(&self, event: &TransactionEvent) -> bool {
        let filter = &self.config.filter;

        // Check program ID filter
        if !filter.program_ids.contains(&event.program_id.to_string()) {
            return false;
        }

        // Check token mint filter if specified
        if let Some(token_mint) = &event.token_mint {
            if !filter.token_mints.contains(&token_mint.to_string()) {
                return false;
            }
        }

        // Check minimum size
        if let Some(amount) = event.amount {
            if amount < filter.min_size {
                return false;
            }
        }

        // Check transaction success
        if !filter.include_failed && !event.success {
            return false;
        }

        true
    }
}

impl Default for TransactionMonitor {
    fn default() -> Self {
        Self::new(ListenBotConfig::default())
    }
}

/// Trait for parsing DEX-specific transactions
#[async_trait]
pub trait DexTransactionParser: std::fmt::Debug {
    /// Parse a transaction to extract DEX-specific information
    async fn parse_transaction(&self, tx_info: TransactionInfo) -> Result<Option<TransactionEvent>>;
} 