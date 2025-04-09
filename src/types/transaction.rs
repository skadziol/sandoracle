use serde::{Deserialize, Serialize};
use solana_sdk::{
    pubkey::Pubkey,
    transaction::Transaction,
};
use solana_transaction_status::UiTransactionStatusMeta;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionInfo {
    pub signature: String,
    pub slot: u64,
    pub block_time: i64,
    pub success: bool,
    pub program_id: String,
    pub token_transfers: Vec<TokenTransfer>,
    pub raw_transaction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTransfer {
    pub mint: String,
    pub from: String,
    pub to: String,
    pub amount: u64,
}

impl TransactionInfo {
    pub fn new(
        signature: String,
        slot: u64,
        block_time: i64,
        success: bool,
        program_id: String,
        token_transfers: Vec<TokenTransfer>,
        raw_transaction: String,
    ) -> Self {
        Self {
            signature,
            slot,
            block_time,
            success,
            program_id,
            token_transfers,
            raw_transaction,
        }
    }

    pub fn get_transfer_amount(&self, mint: &str) -> Option<u64> {
        self.token_transfers
            .iter()
            .find(|transfer| transfer.mint == mint)
            .map(|transfer| transfer.amount)
    }
} 