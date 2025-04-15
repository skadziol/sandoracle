use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use std::fmt;
use crate::listen_bot::transaction::{TransactionInfo, TransactionEvent, DexTransactionParser};
use crate::error::Result;
use tracing;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DexType {
    Orca,
    Raydium,
    Jupiter,
}

impl fmt::Display for DexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DexType::Orca => write!(f, "Orca"),
            DexType::Raydium => write!(f, "Raydium"),
            DexType::Jupiter => write!(f, "Jupiter"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexSwap {
    pub dex: DexType,
    pub input_token: Pubkey,
    pub output_token: Pubkey,
    pub input_amount: u64,
    pub output_amount: u64,
    pub timestamp: i64,
}

pub struct DexParserFactory;

impl DexParserFactory {
    pub fn create_parser(dex_type: DexType) -> Box<dyn DexTransactionParser> {
        match dex_type {
            DexType::Orca => Box::new(OrcaParser),
            DexType::Raydium => Box::new(RaydiumParser),
            DexType::Jupiter => Box::new(JupiterParser),
        }
    }
}

#[derive(Debug)]
struct OrcaParser;

#[async_trait::async_trait]
impl DexTransactionParser for OrcaParser {
    async fn parse_transaction(&self, _tx_info: TransactionInfo) -> Result<Option<TransactionEvent>> {
        // Implement proper parsing logic
        // Example: If a swap is parsed, log it
        // if let Some(event) = parsed_event {
        //     tracing::info!(
        //         "Parsed DEX swap: dex=Orca, signature={}",
        //         tx_info.signature
        //     );
        //     return Ok(Some(event));
        // }
        Ok(None)
    }

    fn dex_name(&self) -> &'static str {
        "Orca"
    }
}

#[derive(Debug)]
struct RaydiumParser;

#[async_trait::async_trait]
impl DexTransactionParser for RaydiumParser {
    async fn parse_transaction(&self, _tx_info: TransactionInfo) -> Result<Option<TransactionEvent>> {
        // Implement proper parsing logic
        // Example: If a swap is parsed, log it
        // if let Some(event) = parsed_event {
        //     tracing::info!(
        //         "Parsed DEX swap: dex=Raydium, signature={}",
        //         tx_info.signature
        //     );
        //     return Ok(Some(event));
        // }
        Ok(None)
    }

    fn dex_name(&self) -> &'static str {
        "Raydium"
    }
}

#[derive(Debug)]
pub struct JupiterParser;

#[async_trait::async_trait]
impl DexTransactionParser for JupiterParser {
    async fn parse_transaction(&self, tx_info: TransactionInfo) -> Result<Option<TransactionEvent>> {
        tracing::debug!(
            "Checking if transaction {} is a Jupiter swap",
            tx_info.signature
        );
        
        // Jupiter mainnet program ID
        const JUPITER_PROGRAM_ID: &str = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4";
        let jupiter_program = Pubkey::from_str_const(JUPITER_PROGRAM_ID);

        // Check if any instruction in the transaction is for Jupiter
        let message = &tx_info.transaction.message;
        let mut found = false;
        for ix in message.instructions().iter() {
            let program_idx = ix.program_id_index as usize;
            if let Some(program_id) = message.static_account_keys().get(program_idx) {
                tracing::debug!("Checking program ID: {}", program_id);
                if program_id == &jupiter_program {
                    found = true;
                    break;
                }
            }
        }
        if !found {
            return Ok(None);
        }

        // Basic example: use the first two account keys as input/output tokens (for demo)
        // In production, parse instruction data and post-token balances for accuracy
        let input_token = message.static_account_keys().get(1).cloned();
        let output_token = message.static_account_keys().get(2).cloned();

        // Build a TransactionEvent for the Jupiter swap
        let event = TransactionEvent {
            signature: tx_info.signature.parse().unwrap_or_default(),
            program_id: jupiter_program,
            token_mint: input_token, // For demo, use input token as token_mint
            amount: tx_info.amount,  // TODO: Parse from instruction data or balances
            success: tx_info.success,
        };
        tracing::info!(
            "Parsed DEX swap: dex=Jupiter, signature={}, input_token={:?}, output_token={:?}, amount={:?}",
            tx_info.signature,
            input_token,
            output_token,
            tx_info.amount
        );
        Ok(Some(event))
    }

    fn dex_name(&self) -> &'static str {
        "Jupiter"
    }
} 