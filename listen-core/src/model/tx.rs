use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use super::token::Token;
use crate::router::dexes::DexName;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapInfo {
    pub token_in: Token,
    pub token_out: Token,
    pub amount_in: u64,
    pub amount_out: u64,
    pub expected_out: Option<u64>,
    pub price_impact: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Transaction signature
    pub signature: String,
    /// Transaction signer (wallet)
    pub signer: Pubkey,
    /// DEX where the swap is occurring
    pub dex: Option<DexName>,
    /// Swap information if this is a swap transaction
    pub swap_info: Option<SwapInfo>,
    /// Block time
    pub block_time: Option<i64>,
    /// Slot number
    pub slot: Option<u64>,
}
