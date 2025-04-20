use serde::{Serialize, Deserialize};
use solana_sdk::{pubkey::Pubkey, instruction::CompiledInstruction, message::{Message, VersionedMessage}};
use borsh::{BorshDeserialize, BorshSerialize};
use tracing::warn;

// TODO: Add other necessary types here as the project grows

/// Holds details extracted from decoding a specific transaction instruction (e.g., a swap).
/// Fields are optional as decoding might fail or instructions might vary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedInstructionInfo {
    // Could add instruction index here if needed
    pub program_id: String,
    pub instruction_name: String, // e.g., "Swap", "Route", "InitializePool"
    
    // Fields common to Swaps
    pub input_mint: Option<String>,
    pub output_mint: Option<String>,
    pub input_account: Option<String>,
    pub output_account: Option<String>,
    pub input_owner: Option<String>,
    pub output_owner: Option<String>,
    pub input_amount: Option<u64>, // Raw amount
    pub minimum_output_amount: Option<u64>, // For slippage
    pub output_amount: Option<u64>, // If fixed output amount

    // Fields common to Pool Initialization
    pub token_a_mint: Option<String>,
    pub token_b_mint: Option<String>,
    pub initial_token_a_amount: Option<u64>,
    pub initial_token_b_amount: Option<u64>,
    pub pool_address: Option<String>, // The newly created pool address
    
    // Add other fields as needed, e.g., authority, specific DEX params
}

/// Represents the most relevant decoded information from a transaction for MEV purposes.
/// This might contain details from the primary instruction (e.g., the main swap in a Jupiter route)
/// or a specific event like pool creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedTransactionDetails {
    pub signature: String,
    // Could contain multiple decoded instructions if needed, but often one is primary
    pub primary_instruction: Option<DecodedInstructionInfo>,
    // Maybe add priority fee info if parsed from ComputeBudget instruction?
    pub priority_fee_details: Option<PriorityFeeDetails>, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityFeeDetails {
    pub compute_unit_limit: Option<u32>,
    pub compute_unit_price_micro_lamports: Option<u64>,
}

// --- Structs for Jupiter V6 Decoding (Simplified) ---

// Represents the fields we want to extract from the `route` instruction data.
// We are skipping the complex `route_plan` for now.
// Needs `BorshDeserialize` to parse the instruction data byte slice.
#[derive(BorshDeserialize, Debug)]
struct RouteArgsSimple {
    // We need to deserialize the route_plan even if we don't use it fully yet,
    // otherwise borsh will fail if the data layout doesn't match exactly.
    // Define a placeholder or the actual complex type if needed later.
    // For now, let's assume we can skip it if borsh allows partial deserialize,
    // but likely we need to define at least a placeholder Vec.
    // UPDATE: Borsh requires the full struct layout. Let's try skipping manually.
    // We know the fixed size args come *after* the Vec<RoutePlanStep>.
    // Let's define the args we care about and parse manually after the Vec.
    
    // Actual fields based on IDL (after route_plan Vec):
    // pub in_amount: u64,
    // pub quoted_out_amount: u64,
    // pub slippage_bps: u16,
    // pub platform_fee_bps: u8,
}

// --- Structs for Orca Whirlpool Decoding --- 

#[derive(BorshDeserialize, Debug)]
struct OrcaSwapArgs {
    pub amount: u64,
    pub other_amount_threshold: u64,
    pub sqrt_price_limit: u128,
    pub amount_specified_is_input: bool,
    pub a_to_b: bool,
}

// --- Structs for Raydium CLMM Decoding --- 

#[derive(BorshDeserialize, Debug)]
struct RaydiumSwapV2Args {
    pub amount: u64,
    pub other_amount_threshold: u64,
    pub sqrt_price_limit_x64: u128,
    pub is_base_input: bool,
    // Note: The IDL might have additional fields if using optional features like V2, 
    // but these are the core swap args.
}

// --- Structs for Jupiter V6 Decoding --- 

// NOTE: This is complex. We might only define variants we actively parse.
// Need to ensure borsh serialization matches exactly.

#[derive(BorshDeserialize, BorshSerialize, Debug, Clone)] // Add BorshSerialize for potential use
pub enum Side {
    Bid,
    Ask,
}

// Define nested enums/structs based on IDL `types` section first
#[derive(BorshDeserialize, BorshSerialize, Debug, Clone)]
pub enum Swap {
    Saber,
    SaberAddDecimalsDeposit,
    SaberAddDecimalsWithdraw,
    TokenSwap,
    Sencha,
    Step,
    Cropper,
    Raydium,
    Crema { a_to_b: bool },
    Lifinity,
    Mercurial,
    Cykura,
    Serum { side: Side },
    MarinadeDeposit,
    MarinadeUnstake,
    Aldrin { side: Side },
    AldrinV2 { side: Side },
    Whirlpool { a_to_b: bool },
    Invariant { x_to_y: bool },
    Meteora,
    GooseFX,
    DeltaFi { stable: bool },
    Balansol,
    MarcoPolo { x_to_y: bool },
    Dradex { side: Side },
    LifinityV2,
    RaydiumClmm,
    Openbook { side: Side },
    Phoenix { side: Side },
    Symmetry { from_token_id: u64, to_token_id: u64 },
    TokenSwapV2,
    HeliumTreasuryManagementRedeemV0,
    StakeDexStakeWrappedSol,
    StakeDexSwapViaStake { bridge_stake_seed: u32 },
    GooseFXV2,
    Perps,
    PerpsAddLiquidity,
    PerpsRemoveLiquidity,
    MeteoraDlmm,
    OpenBookV2 { side: Side },
    RaydiumClmmV2,
    StakeDexPrefundWithdrawStakeAndDepositStake { bridge_stake_seed: u32 },
    Clone { pool_index: u8, quantity_is_input: bool, quantity_is_collateral: bool },
    SanctumS { src_lst_value_calc_accs: u8, dst_lst_value_calc_accs: u8, src_lst_index: u32, dst_lst_index: u32 },
    SanctumSAddLiquidity { lst_value_calc_accs: u8, lst_index: u32 },
    SanctumSRemoveLiquidity { lst_value_calc_accs: u8, lst_index: u32 },
    RaydiumCP,
    // Note: WhirlpoolSwapV2 has RemainingAccountsInfo option - borsh might struggle with Option<Defined>
    // Skipping complex variants like WhirlpoolSwapV2 for now
    // Add other variants as needed based on IDL
}

#[derive(BorshDeserialize, BorshSerialize, Debug, Clone)]
pub struct RoutePlanStep {
    pub swap: Swap,
    pub percent: u8,
    pub input_index: u8,
    pub output_index: u8,
}

// Argument struct for Jupiter V6 `route` instruction
#[derive(BorshDeserialize, Debug)]
struct JupiterRouteArgs {
    pub route_plan: Vec<RoutePlanStep>,
    pub in_amount: u64,
    pub quoted_out_amount: u64,
    pub slippage_bps: u16,
    pub platform_fee_bps: u8,
}

// Placeholder function where real decoding would happen
pub fn try_decode_transaction(
    signature: String, 
    message: &VersionedMessage, 
    instructions: &[CompiledInstruction]
) -> Option<DecodedTransactionDetails> {
    
    const JUPITER_V6_PROGRAM_ID: &str = "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4";
    const JUPITER_ROUTE_DISCRIMINATOR: [u8; 8] = [229, 23, 203, 151, 122, 227, 173, 42];
    
    const ORCA_WHIRLPOOL_PROGRAM_ID: &str = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc";
    const ORCA_SWAP_DISCRIMINATOR: [u8; 8] = [248, 198, 158, 145, 225, 117, 135, 200];
    
    const RAYDIUM_CLMM_PROGRAM_ID: &str = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK";
    const RAYDIUM_SWAPV2_DISCRIMINATOR: [u8; 8] = [137, 167, 103, 9, 189, 210, 137, 20]; // Calculated: sha256("global::swap_v2")[0..8]

    let mut primary_instruction_info: Option<DecodedInstructionInfo> = None;
    let mut fee_details: Option<PriorityFeeDetails> = None;

    for (ix_index, ix) in instructions.iter().enumerate() {
        // Resolve program ID using the message accounts list
        let account_keys = message.static_account_keys(); // Get account keys from VersionedMessage
        let Some(program_pubkey) = account_keys.get(ix.program_id_index as usize) else {
            warn!(target: "tx_decoder", tx_sig=%signature, ix_index, "Invalid program_id_index");
            continue;
        };
        let program_id = program_pubkey.to_string();

        // --- Attempt to Decode Priority Fee Instruction --- 
        // Check if it's a ComputeBudget instruction (common for priority fees)
        if program_id == "ComputeBudget111111111111111111111111111111" {
            // Instruction data format: u8 (instruction index), u32 (units), u64 (price)
            if ix.data.len() == 13 && ix.data[0] == 3 { // SetComputeUnitPrice instruction index is 3
                if let Ok(price_bytes) = ix.data[5..13].try_into() {
                    fee_details = Some(PriorityFeeDetails {
                        compute_unit_limit: None, // Limit is often set by a separate instruction (idx 2)
                        compute_unit_price_micro_lamports: Some(u64::from_le_bytes(price_bytes)),
                    });
                    // Could also parse compute_unit_limit from instruction with data[0]==2
                }
            }
            continue; // Move to next instruction after processing compute budget
        }

        // --- Attempt to Decode Jupiter V6 Route Instruction --- 
        if program_id == JUPITER_V6_PROGRAM_ID && ix.data.starts_with(&JUPITER_ROUTE_DISCRIMINATOR) {
            warn!(target: "tx_decoder", tx_sig=%signature, ix_index, "Found Jupiter V6 Route - attempting borsh deserialize.");
            
            // Use borsh deserialization
            match JupiterRouteArgs::try_from_slice(&ix.data[8..]) {
                Ok(args) => {
                    let minimum_output_amount = args.quoted_out_amount
                        .saturating_mul(10000u64.saturating_sub(args.slippage_bps as u64)) / 10000u64;

                    // Get accounts using VersionedMessage context
                    let source_account_index = ix.accounts.get(2).copied();
                    let destination_account_index = ix.accounts.get(3).copied();
                    let source_token_account = source_account_index
                        .and_then(|idx| account_keys.get(idx as usize).map(|pk| pk.to_string()));
                    let destination_token_account = destination_account_index
                        .and_then(|idx| account_keys.get(idx as usize).map(|pk| pk.to_string()));

                    // TODO: Fetch mints for source/destination accounts
                    let input_mint = Some("JupiterInputMintPlaceholder".to_string()); 
                    let output_mint = Some("JupiterOutputMintPlaceholder".to_string());

                    primary_instruction_info = Some(DecodedInstructionInfo {
                        program_id,
                        instruction_name: "JupiterV6Route".to_string(),
                        input_mint,
                        output_mint,
                        input_account: source_token_account,
                        output_account: destination_token_account,
                        input_owner: None, 
                        output_owner: None, 
                        input_amount: Some(args.in_amount),
                        minimum_output_amount: Some(minimum_output_amount),
                        output_amount: None, // Not specified in this instruction
                        // ... (pool init fields = None) ...
                        token_a_mint: None,
                        token_b_mint: None,
                        initial_token_a_amount: None,
                        initial_token_b_amount: None,
                        pool_address: None,
                    });
                    break; // Found primary instruction
                },
                Err(e) => {
                    warn!(target: "tx_decoder", tx_sig=%signature, ix_index, error=%e, "Failed to borsh deserialize JupiterRouteArgs");
                }
            }
        }

        // --- Attempt to Decode Orca Whirlpool Swap Instruction --- 
        else if program_id == ORCA_WHIRLPOOL_PROGRAM_ID && ix.data.starts_with(&ORCA_SWAP_DISCRIMINATOR) {
            warn!(target: "tx_decoder", tx_sig=%signature, ix_index, "Found Orca Whirlpool Swap instruction - decoding amounts/slippage.");

            match OrcaSwapArgs::try_from_slice(&ix.data[8..]) {
                Ok(args) => {
                    let mut input_amount: Option<u64> = None;
                    let mut minimum_output_amount: Option<u64> = None;
                    // let mut maximum_input_amount: Option<u64> = None; // If needed

                    if args.amount_specified_is_input {
                        input_amount = Some(args.amount);
                        minimum_output_amount = Some(args.other_amount_threshold);
                    } else {
                        // amount is output amount, threshold is max input
                        // maximum_input_amount = Some(args.other_amount_threshold);
                        minimum_output_amount = Some(args.amount); // Report min output received
                    }

                    // Get accounts using VersionedMessage context
                    let token_owner_account_a_idx = ix.accounts.get(3).copied();
                    let token_owner_account_b_idx = ix.accounts.get(5).copied();
                    let token_owner_account_a = token_owner_account_a_idx
                        .and_then(|idx| account_keys.get(idx as usize).map(|pk| pk.to_string()));
                    let token_owner_account_b = token_owner_account_b_idx
                        .and_then(|idx| account_keys.get(idx as usize).map(|pk| pk.to_string()));
                    
                    // Determine input/output accounts based on a_to_b flag
                    let (input_account, output_account) = if args.a_to_b {
                        (token_owner_account_a, token_owner_account_b)
                    } else {
                        (token_owner_account_b, token_owner_account_a)
                    };
                    
                    // TODO: Get mints by fetching account data for input/output accounts or pool state.
                    let input_mint = Some("OrcaInputMintPlaceholder".to_string()); 
                    let output_mint = Some("OrcaOutputMintPlaceholder".to_string());

                    primary_instruction_info = Some(DecodedInstructionInfo {
                        program_id,
                        instruction_name: "OrcaSwap".to_string(),
                        input_mint,
                        output_mint,
                        input_account,
                        output_account,
                        input_owner: None, // Requires fetching account data
                        output_owner: None, // Requires fetching account data
                        input_amount,
                        minimum_output_amount,
                        output_amount: if args.amount_specified_is_input { None } else { Some(args.amount) },
                        // ... (pool init fields = None) ...
                        token_a_mint: None,
                        token_b_mint: None,
                        initial_token_a_amount: None,
                        initial_token_b_amount: None,
                        pool_address: None,
                    });
                    
                    break; // Found primary instruction
                },
                Err(e) => {
                    warn!(target: "tx_decoder", tx_sig=%signature, ix_index, error=%e, "Failed to deserialize OrcaSwapArgs");
                    // Continue searching other instructions
                }
            }
        }
        
        // --- Attempt to Decode Raydium CLMM SwapV2 Instruction --- 
        else if program_id == RAYDIUM_CLMM_PROGRAM_ID && ix.data.starts_with(&RAYDIUM_SWAPV2_DISCRIMINATOR) {
            warn!(target: "tx_decoder", tx_sig=%signature, ix_index, "Found Raydium CLMM SwapV2 instruction - decoding args.");

            match RaydiumSwapV2Args::try_from_slice(&ix.data[8..]) {
                Ok(args) => {
                    let mut input_amount: Option<u64> = None;
                    let mut minimum_output_amount: Option<u64> = None;

                    if args.is_base_input {
                        input_amount = Some(args.amount);
                        minimum_output_amount = Some(args.other_amount_threshold);
                    } else {
                        minimum_output_amount = Some(args.amount);
                    }
                    
                    // Get accounts using VersionedMessage context
                    let input_token_account_idx = ix.accounts.get(3).copied();
                    let output_token_account_idx = ix.accounts.get(4).copied();
                    let input_vault_mint_idx = ix.accounts.get(11).copied(); 
                    let output_vault_mint_idx = ix.accounts.get(12).copied(); 

                    let input_account = input_token_account_idx
                        .and_then(|idx| account_keys.get(idx as usize).map(|pk| pk.to_string()));
                    let output_account = output_token_account_idx
                        .and_then(|idx| account_keys.get(idx as usize).map(|pk| pk.to_string()));
                    let input_mint = input_vault_mint_idx
                        .and_then(|idx| account_keys.get(idx as usize).map(|pk| pk.to_string()));
                    let output_mint = output_vault_mint_idx
                        .and_then(|idx| account_keys.get(idx as usize).map(|pk| pk.to_string()));

                    primary_instruction_info = Some(DecodedInstructionInfo {
                        program_id,
                        instruction_name: "RaydiumSwapV2".to_string(),
                        input_mint,
                        output_mint,
                        input_account,
                        output_account,
                        input_owner: None, 
                        output_owner: None, 
                        input_amount,
                        minimum_output_amount,
                        output_amount: if args.is_base_input { None } else { Some(args.amount) },
                        token_a_mint: None,
                        token_b_mint: None,
                        initial_token_a_amount: None,
                        initial_token_b_amount: None,
                        pool_address: None,
                    });
                    
                    break; // Found primary instruction
                },
                Err(e) => {
                    warn!(target: "tx_decoder", tx_sig=%signature, ix_index, error=%e, "Failed to deserialize RaydiumSwapV2Args");
                }
            }
        }
        
        // TODO: Add checks for other program IDs (Raydium, etc.)
    }

    // Return details if we decoded something relevant
    if primary_instruction_info.is_some() || fee_details.is_some() {
        Some(DecodedTransactionDetails {
            signature,
            primary_instruction: primary_instruction_info,
            priority_fee_details: fee_details,
        })
    } else {
        // No relevant instruction decoded
        None
    }
} 