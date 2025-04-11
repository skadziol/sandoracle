use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use super::dexes::DexName;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteResponse {
    /// Input token mint
    pub token_in: Pubkey,
    /// Output token mint
    pub token_out: Pubkey,
    /// Amount of input token
    pub amount_in: u64,
    /// Expected amount of output token
    pub amount_out: u64,
    /// Minimum amount out (accounting for slippage)
    pub minimum_out: u64,
    /// Price impact percentage
    pub price_impact: f64,
    /// DEX providing the quote
    pub dex: DexName,
    /// Fee in basis points
    pub fee_bps: u16,
}
