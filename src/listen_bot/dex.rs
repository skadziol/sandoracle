use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use std::fmt;
use crate::listen_bot::transaction::{TransactionInfo, TransactionEvent, DexTransactionParser, DexTransaction, TokenAmount};
use crate::error::Result;
use tracing::{self, warn, debug, trace, info};
use borsh::{BorshDeserialize, BorshSerialize};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use solana_sdk::instruction::InstructionError;
use solana_transaction_status::EncodedConfirmedTransactionWithStatusMeta;
use std::collections::HashMap;
use std::str::FromStr;

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
    async fn parse_transaction(&self, _tx_info: TransactionInfo) -> Result<Option<DexTransaction>> {
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
    async fn parse_transaction(&self, _tx_info: TransactionInfo) -> Result<Option<DexTransaction>> {
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

// Define the assumed structure for the Jupiter swap event
#[derive(BorshDeserialize, BorshSerialize, Debug, Clone)]
pub struct JupiterSwapEventData { // Renamed to avoid conflict if Anchor adds its own Event struct
    pub input_mint: Pubkey,
    pub output_mint: Pubkey,
    pub input_amount: u64,
    pub output_amount: u64,
    // Add other potential fields if needed based on actual IDL later
    // pub user: Pubkey, 
    // pub slippage_bps: u16, 
}

#[derive(Debug)]
pub struct JupiterParser;

#[async_trait::async_trait]
impl DexTransactionParser for JupiterParser {
    async fn parse_transaction(&self, tx_info: TransactionInfo) -> Result<Option<DexTransaction>> { // Changed return type
        // Jupiter mainnet program ID
        const JUPITER_PROGRAM_ID: &str = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4";
        let jupiter_program_pk = Pubkey::from_str(JUPITER_PROGRAM_ID).unwrap();

        // Check if the transaction interacted with Jupiter V6
        let interacts_with_jupiter = tx_info
            .transaction
            .message
            .static_account_keys()
            .iter()
            .any(|key| key == &jupiter_program_pk)
            || tx_info.transaction.message.instructions().iter().any(|ix| {
                let program_idx = ix.program_id_index as usize;
                tx_info.transaction.message.static_account_keys().get(program_idx) == Some(&jupiter_program_pk)
            });

        if !interacts_with_jupiter {
            return Ok(None); // Not a Jupiter transaction
        }
        
        // Check transaction success
        let meta = match tx_info.meta {
            Some(m) => m,
            None => {
                warn!(signature = %tx_info.signature, "Missing metadata for potential Jupiter tx");
                return Ok(None);
            }
        };

        if meta.err.is_some() {
            debug!(signature = %tx_info.signature, "Skipping failed Jupiter tx");
            return Ok(None); // Skip failed transactions for now
        }
        
        // Parse logs for Anchor events
        let log_messages = meta.log_messages.as_ref().map(|v| v.as_slice()).unwrap_or(&[]);
        let mut parsed_event_data: Option<JupiterSwapEventData> = None;

        for log in log_messages {
            if log.starts_with(format!("Program {}", JUPITER_PROGRAM_ID).as_str()) {
                // Find the Anchor event log line
                if let Some(event_log) = log_messages.iter().find(|msg| msg.starts_with("Program log: Anchor event:")) {
                    if let Some(base64_data) = event_log.strip_prefix("Program log: Anchor event: ") {
                        match BASE64_STANDARD.decode(base64_data) {
                            Ok(event_bytes) => {
                                // Anchor events have an 8-byte discriminator prepended
                                if event_bytes.len() > 8 {
                                    // Attempt to deserialize using Borsh, skipping the discriminator
                                    // NOTE: This ASSUMES the event name hash matches what BorshDeserialize expects.
                                    // Using AnchorDeserialize::try_from_slice is safer if we have the full event type.
                                    match JupiterSwapEventData::deserialize(&mut &event_bytes[8..]) {
                                        Ok(event_data) => {
                                            parsed_event_data = Some(event_data);
                                            // Found the event, no need to process more logs for this tx
                                            // Potentially break if multiple swaps can occur? For now, take first.
                                            break; 
                                        }
                                        Err(e) => {
                                            // This likely means our assumed struct doesn't match the actual event layout
                                            // or it's a different Jupiter event.
                                            trace!(signature = %tx_info.signature, error = %e, "Failed Borsh deserialize Jupiter event");
                                        }
                                    }
                                } else {
                                    trace!(signature = %tx_info.signature, "Event bytes too short");
                                }
                            }
                            Err(e) => {
                                trace!(signature = %tx_info.signature, error = %e, "Failed base64 decode Jupiter event");
                            }
                        }
                    }
                }
            }
            // Stop checking logs once we find a potential event (or finish iterating)
            if parsed_event_data.is_some() { break; }
        }

        // If we parsed event data, construct the DexTransaction
        if let Some(event_data) = parsed_event_data {
            let dex_tx = DexTransaction {
                signature: tx_info.signature.parse().unwrap_or_default(),
                program_id: jupiter_program_pk,
                input_token: TokenAmount {
                    mint: event_data.input_mint.to_string(),
                    amount: event_data.input_amount,
                    decimals: 0, // TODO: Fetch decimals from mint info or token balances
                },
                output_token: TokenAmount {
                    mint: event_data.output_mint.to_string(),
                    amount: event_data.output_amount,
                    decimals: 0, // TODO: Fetch decimals from mint info or token balances
                },
                dex_name: self.dex_name().to_string(),
                timestamp: chrono::Utc::now().timestamp(), // TODO: Use block time from meta if available
                succeeded: true, // We checked meta.err above
                fee: meta.fee,
                slippage: 0.0, // TODO: Calculate slippage if possible (needs pre/post balances or quote info)
                dex_metadata: HashMap::new(), // Can add more details here later
                dex_type: DexType::Jupiter, // Set the enum type
            };
             info!(signature = %tx_info.signature, input_mint = %dex_tx.input_token.mint, output_mint = %dex_tx.output_token.mint, input_amount = dex_tx.input_token.amount, output_amount = dex_tx.output_token.amount, "Parsed Jupiter swap event");
            Ok(Some(dex_tx))
        } else {
            // No relevant event found in logs
            Ok(None)
        }
    }

    fn dex_name(&self) -> &'static str {
        "Jupiter"
    }
} 