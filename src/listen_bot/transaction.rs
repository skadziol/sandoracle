use crate::error::{Result, SandoError, TransactionErrorKind};
use crate::listen_bot::dex::DexType;
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
use tracing::{debug};

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

        // Check if transaction is from a monitored program
        let program_id_str = event.program_id.to_string();
        if !filter.program_ids.iter().any(|id| id.to_string() == program_id_str) {
            debug!("Skipping transaction: program_id not monitored");
            return false;
        }

        // If the transaction involves tokens, check if any monitored tokens are involved
        if let Some(token_mint) = &event.token_mint {
            let token_mint_str = token_mint.to_string();
            if !filter.token_mints.iter().any(|mint| mint.to_string() == token_mint_str) {
                debug!("Skipping transaction: token_mint not monitored");
                return false;
            }
        }

        // Check minimum transaction size if applicable
        if let Some(amount) = event.amount {
            if filter.min_size > 0 && amount < filter.min_size {
                debug!("Skipping transaction: amount {} below minimum {}", amount, filter.min_size);
                return false;
            }
        }

        // Check if we should include failed transactions
        if !event.success && !filter.include_failed {
            debug!("Skipping failed transaction");
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
pub trait DexTransactionParser: Send + Sync + std::fmt::Debug {
    fn dex_name(&self) -> &'static str;
    async fn parse_transaction(&self, tx_info: TransactionInfo) -> Result<Option<TransactionEvent>>;
}

#[derive(Debug, Clone)]
pub struct TransactionInfo {
    pub transaction: Transaction,
    pub meta: Option<UiTransactionStatusMeta>,
    pub signature: String,
    pub program_id: String,
    pub token_mint: Option<String>,
    pub amount: Option<u64>,
    pub success: bool,
} 