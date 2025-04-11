use std::str::FromStr;

use anyhow::{anyhow, Result};
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use solana_sdk::transaction::VersionedTransaction;
// Add bincode use statement if not already present transitively
// use bincode;
// Add reqwest use statement
// use reqwest;

#[derive(Serialize, Deserialize, Debug, Clone)] // Added Clone
pub struct PlatformFee {
    pub amount: String,
    #[serde(rename = "feeBps")]
    pub fee_bps: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)] // Added Clone
pub struct DynamicSlippage {
    #[serde(rename = "minBps")]
    pub min_bps: i32,
    #[serde(rename = "maxBps")]
    pub max_bps: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)] // Added Clone
pub struct SwapInfo {
    #[serde(rename = "ammKey")]
    pub amm_key: String,
    pub label: Option<String>,
    #[serde(rename = "inputMint")]
    pub input_mint: String,
    #[serde(rename = "outputMint")]
    pub output_mint: String,
    #[serde(rename = "inAmount")]
    pub in_amount: String,
    #[serde(rename = "outAmount")]
    pub out_amount: String,
    #[serde(rename = "feeAmount")]
    pub fee_amount: String,
    #[serde(rename = "feeMint")]
    pub fee_mint: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)] // Added Clone
pub struct RoutePlan {
    #[serde(rename = "swapInfo")]
    pub swap_info: SwapInfo,
    pub percent: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone)] // Added Clone
pub struct QuoteResponse {
    #[serde(rename = "inputMint")]
    pub input_mint: String,
    #[serde(rename = "inAmount")]
    pub in_amount: String,
    #[serde(rename = "outputMint")]
    pub output_mint: String,
    #[serde(rename = "outAmount")]
    pub out_amount: String,
    #[serde(rename = "otherAmountThreshold")]
    pub other_amount_threshold: String,
    #[serde(rename = "swapMode")]
    pub swap_mode: String,
    #[serde(rename = "slippageBps")]
    pub slippage_bps: i32,
    #[serde(rename = "platformFee")]
    pub platform_fee: Option<PlatformFee>,
    #[serde(rename = "priceImpactPct")]
    pub price_impact_pct: String,
    #[serde(rename = "routePlan")]
    pub route_plan: Vec<RoutePlan>,
    #[serde(rename = "contextSlot")]
    pub context_slot: u64,
    #[serde(rename = "timeTaken")]
    pub time_taken: f64,
}


// SwapRequest is used internally by Jupiter::swap, doesn't need to be pub
#[derive(Serialize)]
struct SwapRequest {
    #[serde(rename = "userPublicKey")]
    user_public_key: String,
    #[serde(rename = "wrapAndUnwrapSol")]
    wrap_and_unwrap_sol: bool, // Default false is fine
    #[serde(rename = "useSharedAccounts")]
    use_shared_accounts: bool, // Default false is fine
    #[serde(rename = "feeAccount")]
    fee_account: Option<String>,
    #[serde(rename = "trackingAccount")]
    tracking_account: Option<String>,
    #[serde(rename = "computeUnitPriceMicroLamports")]
    compute_unit_price_micro_lamports: Option<u64>,
    #[serde(rename = "prioritizationFeeLamports")]
    prioritization_fee_lamports: Option<u64>,
    #[serde(rename = "asLegacyTransaction")]
    as_legacy_transaction: bool, // Default false is fine
    #[serde(rename = "useTokenLedger")]
    use_token_ledger: bool, // Default false is fine
    #[serde(rename = "destinationTokenAccount")]
    destination_token_account: Option<String>,
    #[serde(rename = "dynamicComputeUnitLimit")]
    dynamic_compute_unit_limit: bool, // Default false is fine
    #[serde(rename = "skipUserAccountsRpcCalls")]
    skip_user_accounts_rpc_calls: bool, // Default false is fine
    #[serde(rename = "dynamicSlippage")]
    dynamic_slippage: Option<DynamicSlippage>, // Can be None
    #[serde(rename = "quoteResponse")]
    quote_response: QuoteResponse,
}

// SwapResponse is used internally by Jupiter::swap
#[derive(Deserialize, Debug)]
struct SwapResponse {
    #[serde(rename = "swapTransaction")]
    swap_transaction: String,
    // last_valid_block_height: u64, // Not needed for deserialization
}

// InstructionData and AccountMeta are used internally by _convert_instruction_data
#[derive(Deserialize, Debug)]
struct InstructionData {
    #[serde(rename = "programId")]
    program_id: String,
    accounts: Vec<AccountMeta>,
    data: String,
}

#[derive(Deserialize, Debug)]
struct AccountMeta {
    pubkey: String,
    #[serde(rename = "isSigner")]
    is_signer: bool,
    #[serde(rename = "isWritable")]
    is_writable: bool,
}

// Make Jupiter struct public
pub struct Jupiter;

impl Jupiter {
    // Make fetch_quote public
    pub async fn fetch_quote(
        input_mint: &str,
        output_mint: &str,
        amount: u64,
    ) -> Result<QuoteResponse> {
        let url = format!(
            "https://quote-api.jup.ag/v6/quote?inputMint={}&outputMint={}&amount={}",
            input_mint, output_mint, amount,
        );
        // Ensure reqwest is available
        let response = reqwest::get(&url).await?.json::<QuoteResponse>().await?;
        Ok(response)
    }

    // Make swap public
    pub async fn swap(
        quote_response: QuoteResponse,
        owner: &Pubkey,
    ) -> Result<VersionedTransaction> {
        let swap_request = SwapRequest {
            user_public_key: owner.to_string(),
            quote_response,
            wrap_and_unwrap_sol: true, // Wrap/unwrap SOL automatically
            use_shared_accounts: true, // Use shared accounts to potentially save fees
            dynamic_slippage: None, // Let Jupiter handle slippage based on quote
            // Set other fields to default/None as needed
            fee_account: None,
            tracking_account: None,
            compute_unit_price_micro_lamports: None,
            prioritization_fee_lamports: None,
            as_legacy_transaction: false,
            use_token_ledger: false,
            destination_token_account: None,
            dynamic_compute_unit_limit: true,
            skip_user_accounts_rpc_calls: true,
        };
        let client = reqwest::Client::new();
        let raw_res = client
            .post("https://quote-api.jup.ag/v6/swap")
            .json(&swap_request)
            .send()
            .await?;

        if !raw_res.status().is_success() {
            let error_body = raw_res.text().await.unwrap_or_else(|_| "Unknown error body".to_string());
            return Err(anyhow!("Jupiter swap API failed: {}", error_body));
        }
        let response = raw_res
            .json::<SwapResponse>()
            .await
            .map_err(|e| anyhow!("Failed to parse Jupiter swap response: {}", e))?;

        // Ensure bincode and base64 are available
        let decoded_tx = BASE64_STANDARD
            .decode(response.swap_transaction)
            .map_err(|e| anyhow!("Failed to decode base64 swap transaction: {}", e))?;
        let tx: VersionedTransaction = bincode::deserialize(&decoded_tx)
            .map_err(|e| anyhow!("Failed to deserialize transaction: {}", e))?;

        Ok(tx)
    }

    // _convert_instruction_data is internal helper, keep private
    #[allow(dead_code)] // Allow dead code for now as it's not used directly
    fn _convert_instruction_data(
        ix_data: InstructionData,
    ) -> Result<solana_sdk::instruction::Instruction> {
        let program_id = Pubkey::from_str(&ix_data.program_id)?;

        let accounts = ix_data
            .accounts
            .into_iter()
            .map(|acc| {
                Ok(solana_sdk::instruction::AccountMeta {
                    pubkey: Pubkey::from_str(&acc.pubkey)?,
                    is_signer: acc.is_signer,
                    is_writable: acc.is_writable,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let data = BASE64_STANDARD.decode(ix_data.data)?;

        Ok(solana_sdk::instruction::Instruction {
            program_id,
            accounts,
            data,
        })
    }
}