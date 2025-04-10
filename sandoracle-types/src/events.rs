use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Definition moved from sandoseer/src/listen_bot/transaction.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionEvent {
    pub pipeline_id: Uuid,
    pub step_id: Uuid,
    pub transaction_hash: String,
    pub status: String, // e.g., "Executed", "Failed", "Simulated"
    pub details: String, // e.g., Error message or success details
    pub timestamp: DateTime<Utc>,
} 