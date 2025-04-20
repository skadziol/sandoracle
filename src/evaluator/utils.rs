use tracing::{warn, debug};
use solana_client::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use std::sync::Arc;
use crate::error::{Result as SandoResult, SandoError};
use tokio::task; // For spawn_blocking

/// Calculates the estimated price impact percentage for a trade in an AMM pool.
///
/// This uses a simplified formula suitable for constant product (x*y=k) pools.
/// It will NOT be accurate for concentrated liquidity pools (e.g., Orca Whirlpools).
/// 
/// Args:
/// * `trade_size_usd`: The estimated USD value of the trade (input or output).
/// * `pool_liquidity_usd`: The estimated total USD value locked in the pool.
///
/// Returns:
/// The estimated price impact as a percentage (e.g., 1.5 for 1.5%).
pub fn calculate_price_impact(trade_size_usd: f64, pool_liquidity_usd: f64) -> f64 {
    if pool_liquidity_usd <= 0.0 {
        warn!(target: "evaluator_utils", "Pool liquidity is zero or negative, cannot calculate impact.");
        return 100.0; // Return high impact if liquidity is invalid
    }

    // trade_ratio represents the trade size as a fraction of total pool liquidity
    let trade_ratio = trade_size_usd / pool_liquidity_usd;

    // Using a common approximation for price impact in x*y=k pools:
    // impact = (trade_ratio / (2 - trade_ratio)) * 100
    // We cap trade_ratio at 1.0 to avoid division by zero or negative results if trade > liquidity
    let impact_percentage = (trade_ratio / (2.0 - trade_ratio.min(1.0))) * 100.0;

    // Ensure impact is not negative (can happen with floating point issues on tiny ratios)
    impact_percentage.max(0.0)
}

/// Calculates the net profit after deducting estimated gas costs.
///
/// Args:
/// * `gross_profit_usd`: The estimated profit before costs.
/// * `estimated_gas_cost_usd`: The estimated gas cost in USD.
///
/// Returns:
/// The net profit in USD.
pub fn calculate_net_profit(gross_profit_usd: f64, estimated_gas_cost_usd: f64) -> f64 {
    gross_profit_usd - estimated_gas_cost_usd
}

/// Enum to represent desired priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorityLevel {
    Low,       // e.g., 25th percentile
    Medium,    // e.g., 50th percentile (median)
    High,      // e.g., 75th percentile
    VeryHigh,  // e.g., 95th percentile
    Max,       // Maximum observed
}

/// Estimates the total transaction cost in USD.
///
/// Fetches recent priority fees and combines with base fee.
/// Requires estimated compute units and desired priority level.
///
/// Args:
/// * `rpc_client`: Shared RPC client to fetch fees.
/// * `estimated_compute_units`: Estimated CUs the transaction will consume.
/// * `priority_level`: The desired fee level relative to recent fees.
/// * `sol_price_usd`: Current price of SOL in USD.
/// * `involved_accounts`: Optional list of accounts involved to get targeted fees.
///
/// Returns:
/// SandoResult containing the estimated cost in USD.
pub async fn estimate_transaction_cost_usd(
    rpc_client: Arc<RpcClient>,
    estimated_compute_units: u64,
    priority_level: PriorityLevel,
    sol_price_usd: f64,
    involved_accounts: Option<&[Pubkey]> 
) -> SandoResult<f64> {
    
    const BASE_FEE_LAMPORTS: u64 = 5000; // Base fee per signature
    const DEFAULT_PRIORITY_FEE_PER_CU: u64 = 1000; // Default if RPC fails (microlamports)

    if sol_price_usd <= 0.0 {
        // Use a specific error variant if available, otherwise InternalError
        return Err(SandoError::InternalError("Invalid SOL price for fee estimation".to_string())); 
    }

    // 1. Fetch recent priority fees using spawn_blocking for the sync RpcClient method
    let accounts_slice: Vec<Pubkey> = involved_accounts.unwrap_or(&[]).to_vec(); // Clone pubkeys for sending to blocking task
    let rpc_client_clone = rpc_client.clone(); // Clone Arc for blocking task
    let recent_fees_result = task::spawn_blocking(move || {
         rpc_client_clone.get_recent_prioritization_fees(&accounts_slice) // Pass slice
    }).await;

    let recent_fees = match recent_fees_result {
        Ok(Ok(fees)) => fees,
        Ok(Err(rpc_err)) => {
            return Err(SandoError::SolanaRpc(format!("Failed to fetch recent priority fees: {}", rpc_err)));
        }
        Err(join_err) => {
             return Err(SandoError::InternalError(format!("Spawn blocking task failed for fee fetch: {}", join_err)));
        }
    };

    // 2. Select fee based on priority level
    let mut target_fee_per_cu = DEFAULT_PRIORITY_FEE_PER_CU;
    if !recent_fees.is_empty() {
        let recent_fees_microlamports: Vec<u64> = recent_fees
            .iter()
            .map(|f| f.prioritization_fee)
            .collect();

        if !recent_fees_microlamports.is_empty() {
            let mut sorted_fees = recent_fees_microlamports.clone();
            sorted_fees.sort_unstable();
            let n = sorted_fees.len();

            target_fee_per_cu = match priority_level {
                PriorityLevel::Low => sorted_fees.get(n / 4).copied().unwrap_or(DEFAULT_PRIORITY_FEE_PER_CU),
                PriorityLevel::Medium => sorted_fees.get(n / 2).copied().unwrap_or(DEFAULT_PRIORITY_FEE_PER_CU),
                PriorityLevel::High => sorted_fees.get(3 * n / 4).copied().unwrap_or(DEFAULT_PRIORITY_FEE_PER_CU),
                PriorityLevel::VeryHigh => sorted_fees.get(95 * n / 100).copied().unwrap_or(DEFAULT_PRIORITY_FEE_PER_CU),
                PriorityLevel::Max => sorted_fees.last().copied().unwrap_or(DEFAULT_PRIORITY_FEE_PER_CU),
            };
            debug!(target: "fee_estimator", selected_fee_per_cu_microlamports = target_fee_per_cu, ?priority_level, "Selected priority fee");
        } else {
             warn!(target: "fee_estimator", "Received empty fee list from RPC, using default priority fee.");
        }
    } else {
        warn!(target: "fee_estimator", "Received no recent fees from RPC, using default priority fee.");
    }
    
    // 3. Calculate total priority fee lamports
    let total_priority_fee_lamports = (target_fee_per_cu as u128 * estimated_compute_units as u128 / 1_000_000u128) as u64;

    // 4. Calculate total cost in lamports
    let total_cost_lamports = BASE_FEE_LAMPORTS.saturating_add(total_priority_fee_lamports);

    // 5. Convert to USD
    let cost_sol = total_cost_lamports as f64 / 1_000_000_000.0;
    let cost_usd = cost_sol * sol_price_usd;

    debug!(target: "fee_estimator", 
           base_fee = BASE_FEE_LAMPORTS, 
           priority_fee = total_priority_fee_lamports, 
           total_lamports = total_cost_lamports, 
           sol_price = sol_price_usd, 
           cost_usd = cost_usd, 
           "Estimated transaction cost");

    Ok(cost_usd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_price_impact() {
        // Test case 1: Small trade, large liquidity
        let impact1 = calculate_price_impact(1000.0, 1_000_000.0); // 0.1% of liquidity
        assert!((impact1 - 0.1).abs() < 0.01, "Impact was {}", impact1); // ~0.1%

        // Test case 2: Moderate trade
        let impact2 = calculate_price_impact(50_000.0, 1_000_000.0); // 5% of liquidity
        assert!((impact2 - 5.1).abs() < 0.1, "Impact was {}", impact2); // ~5.1%

        // Test case 3: Large trade
        let impact3 = calculate_price_impact(200_000.0, 1_000_000.0); // 20% of liquidity
        assert!((impact3 - 25.0).abs() < 0.1, "Impact was {}", impact3); // ~25%

        // Test case 4: Trade equals liquidity
        let impact4 = calculate_price_impact(1_000_000.0, 1_000_000.0); // 100% of liquidity
        assert!((impact4 - 100.0).abs() < 0.1, "Impact was {}", impact4); // 100%

        // Test case 5: Zero liquidity
        let impact5 = calculate_price_impact(1000.0, 0.0);
        assert!((impact5 - 100.0).abs() < 0.1, "Impact was {}", impact5); // Should return high impact
        
        // Test case 6: Zero trade size
        let impact6 = calculate_price_impact(0.0, 100000.0);
         assert!((impact6 - 0.0).abs() < 0.001, "Impact was {}", impact6); // Should be zero
    }

    #[test]
    fn test_calculate_net_profit() {
        assert_eq!(calculate_net_profit(100.0, 20.0), 80.0);
        assert_eq!(calculate_net_profit(50.0, 50.0), 0.0);
        assert_eq!(calculate_net_profit(30.0, 40.0), -10.0);
        assert_eq!(calculate_net_profit(0.0, 10.0), -10.0);
    }

    // TODO: Add tests for estimate_transaction_cost_usd
} 