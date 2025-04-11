use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// Token mint address
    pub mint: Pubkey,
    /// Token symbol (e.g., "SOL", "USDC")
    pub symbol: Option<String>,
    /// Token decimals
    pub decimals: u8,
    /// Current price in USD (if available)
    pub price_usd: Option<f64>,
}
