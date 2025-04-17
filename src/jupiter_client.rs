use std::str::FromStr;

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use solana_sdk::transaction::VersionedTransaction;
// Add bincode use statement if not already present transitively
// use bincode;
use reqwest;

/// Information about a Solana token from Jupiter's API
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenInfo {
    pub address: String,
    pub chainId: i64,
    pub decimals: i32,
    pub name: String,
    pub symbol: String,
    pub logoURI: Option<String>,
    pub tags: Option<Vec<String>>,
    pub coingeckoId: Option<String>,
}

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
#[derive(Clone)]
pub struct Jupiter;

impl Jupiter {
    /// Create a new Jupiter API client
    pub fn new(api_url: &str) -> Self {
        // For now, ignoring the api_url param since we're using hardcoded URLs
        // In a future version, we could use this to configure the API endpoints
        Jupiter {}
    }

    /// Get price of token in terms of reference token
    pub async fn get_price(&self, token_mint: &str, reference_mint: &str) -> Result<f64> {
        // Use a standard amount for quote (1 token with 9 decimals = 1_000_000_000)
        let amount = 1_000_000_000;
        
        // Fetch the quote from Jupiter API
        let quote = Self::fetch_quote(token_mint, reference_mint, amount).await?;
        
        // Calculate price from the quote
        let in_amount = quote.in_amount.parse::<f64>().map_err(|e| 
            anyhow!("Failed to parse input amount '{}': {}", quote.in_amount, e))?;
        let out_amount = quote.out_amount.parse::<f64>().map_err(|e| 
            anyhow!("Failed to parse output amount '{}': {}", quote.out_amount, e))?;
        
        // Price is output amount / input amount
        let price = out_amount / in_amount;
        
        Ok(price)
    }
    
    // Public method to get token list from Jupiter
    pub async fn get_token_list() -> Result<Vec<TokenInfo>> {
        let url = "https://token.jup.ag/all";
        let response = reqwest::get(url).await?.json::<Vec<TokenInfo>>().await?;
        Ok(response)
    }

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

    // Add a new method to get individual swap instructions
    pub async fn get_swap_instructions(
        quote_response: QuoteResponse,
        owner: &Pubkey,
        min_out_amount: u64, // Allow explicit slippage control
    ) -> Result<Vec<solana_sdk::instruction::Instruction>> {
        // Create a modified quote response with the specified min_out_amount
        let mut quote = quote_response.clone();
        quote.other_amount_threshold = min_out_amount.to_string();

        // Prepare the request for instructions
        let swap_request = SwapRequest {
            user_public_key: owner.to_string(),
            quote_response: quote,
            wrap_and_unwrap_sol: true,
            use_shared_accounts: true,
            fee_account: None,
            tracking_account: None,
            compute_unit_price_micro_lamports: None,
            prioritization_fee_lamports: None,
            as_legacy_transaction: true, // Using legacy transaction format
            use_token_ledger: false,
            destination_token_account: None,
            dynamic_compute_unit_limit: true,
            skip_user_accounts_rpc_calls: true,
            dynamic_slippage: None,
        };

        // Use the Jupiter instructions endpoint
        let client = reqwest::Client::new();
        let raw_res = client
            .post("https://quote-api.jup.ag/v6/swap-instructions")
            .json(&swap_request)
            .send()
            .await?;

        if !raw_res.status().is_success() {
            let error_body = raw_res.text().await.unwrap_or_else(|_| "Unknown error body".to_string());
            return Err(anyhow!("Jupiter swap instructions API failed: {}", error_body));
        }

        // Instructions response from Jupiter
        #[derive(Deserialize, Debug)]
        struct InstructionsResponse {
            #[serde(rename = "tokenLedgerInstruction")]
            token_ledger_instruction: Option<InstructionData>,
            #[serde(rename = "computeBudgetInstructions")]
            compute_budget_instructions: Vec<InstructionData>,
            #[serde(rename = "setupInstructions")]
            setup_instructions: Vec<InstructionData>,
            #[serde(rename = "swapInstruction")]
            swap_instruction: InstructionData,
            #[serde(rename = "cleanupInstruction")]
            cleanup_instruction: Option<InstructionData>,
            #[serde(rename = "addressLookupTableAddresses")]
            address_lookup_table_addresses: Vec<String>,
        }

        // Parse the response
        let instructions_response: InstructionsResponse = raw_res.json().await?;

        // Convert all instruction data to Solana SDK instructions
        let mut instructions = Vec::new();

        // Process compute budget instructions (typically sets compute unit limit and price)
        for ix_data in instructions_response.compute_budget_instructions {
            let instruction = Self::_convert_instruction_data(ix_data)?;
            instructions.push(instruction);
        }

        // Process setup instructions (token account creation, wrapping SOL, etc.)
        for ix_data in instructions_response.setup_instructions {
            let instruction = Self::_convert_instruction_data(ix_data)?;
            instructions.push(instruction);
        }

        // Add token ledger instruction if present
        if let Some(token_ledger) = instructions_response.token_ledger_instruction {
            let instruction = Self::_convert_instruction_data(token_ledger)?;
            instructions.push(instruction);
        }

        // Add the main swap instruction
        let swap_ix = Self::_convert_instruction_data(instructions_response.swap_instruction)?;
        instructions.push(swap_ix);

        // Add cleanup instruction if present (typically for unwrapping SOL)
        if let Some(cleanup) = instructions_response.cleanup_instruction {
            let instruction = Self::_convert_instruction_data(cleanup)?;
            instructions.push(instruction);
        }

        Ok(instructions)
    }
}