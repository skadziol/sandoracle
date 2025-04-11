use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use solana_client::rpc_response::RpcConfirmedTransactionStatusWithSignature;
use solana_sdk::pubkey::Pubkey;
use std::str::FromStr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub signature: String,
    pub block_time: i64,
    pub success: bool,
    pub program_id: Pubkey,
    pub accounts: Vec<Pubkey>,
    pub data: Vec<u8>,
}

impl Transaction {
    pub fn from_rpc_response(response: RpcConfirmedTransactionStatusWithSignature) -> Result<Self> {
        let meta = response.transaction.meta
            .ok_or_else(|| anyhow!("Transaction metadata not found"))?;
        
        let transaction = response.transaction.transaction
            .decode()
            .ok_or_else(|| anyhow!("Failed to decode transaction"))?;

        let instruction = transaction.message.instructions.get(0)
            .ok_or_else(|| anyhow!("No instructions found in transaction"))?;

        Ok(Self {
            signature: response.transaction.signatures[0].to_string(),
            block_time: response.block_time.unwrap_or(0),
            success: meta.err.is_none(),
            program_id: transaction.message.account_keys[instruction.program_id_index as usize],
            accounts: instruction.accounts.iter()
                .map(|&idx| transaction.message.account_keys[idx as usize])
                .collect(),
            data: instruction.data.clone(),
        })
    }
} 