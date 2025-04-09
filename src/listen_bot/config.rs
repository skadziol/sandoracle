use serde::{Deserialize, Serialize};
use solana_sdk::pubkey::Pubkey;
use std::collections::HashSet;

/// Configuration for transaction filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionFilter {
    /// Program IDs to monitor
    pub program_ids: HashSet<Pubkey>,
    
    /// Token mints to monitor
    pub token_mints: HashSet<Pubkey>,
    
    /// Minimum transaction size in lamports
    pub min_size: u64,
    
    /// Whether to include failed transactions
    pub include_failed: bool,
}

impl Default for TransactionFilter {
    fn default() -> Self {
        Self {
            program_ids: HashSet::new(),
            token_mints: HashSet::new(),
            min_size: 0,
            include_failed: false,
        }
    }
}

/// Configuration for the ListenBot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListenBotConfig {
    /// Solana RPC URL
    pub rpc_url: String,
    /// WebSocket URL for real-time updates
    pub ws_url: String,
    /// Transaction filtering configuration
    pub filter: TransactionFilter,
    /// Maximum number of concurrent requests
    pub max_concurrent_requests: usize,
    /// Request timeout in seconds
    pub request_timeout: u64,
    /// Maximum retries for failed requests
    pub max_retries: u32,
    /// Retry backoff base in milliseconds
    pub retry_backoff_ms: u64,
}

impl Default for ListenBotConfig {
    fn default() -> Self {
        Self {
            rpc_url: "https://api.mainnet-beta.solana.com".to_string(),
            ws_url: "wss://api.mainnet-beta.solana.com".to_string(),
            filter: TransactionFilter::default(),
            max_concurrent_requests: 10,
            request_timeout: 30,
            max_retries: 3,
            retry_backoff_ms: 1000,
        }
    }
}

impl ListenBotConfig {
    /// Creates a new configuration with custom RPC and WebSocket URLs
    pub fn new(rpc_url: impl Into<String>, ws_url: impl Into<String>) -> Self {
        Self {
            rpc_url: rpc_url.into(),
            ws_url: ws_url.into(),
            ..Default::default()
        }
    }

    /// Sets the transaction filter configuration
    pub fn with_filter(mut self, filter: TransactionFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Adds program IDs to monitor
    pub fn with_program_ids(mut self, program_ids: impl IntoIterator<Item = Pubkey>) -> Self {
        self.filter.program_ids.extend(program_ids);
        self
    }

    /// Adds token mints to monitor
    pub fn with_token_mints(mut self, token_mints: impl IntoIterator<Item = Pubkey>) -> Self {
        self.filter.token_mints.extend(token_mints);
        self
    }

    /// Sets the minimum transaction size in lamports
    pub fn with_min_size(mut self, min_size: u64) -> Self {
        self.filter.min_size = min_size;
        self
    }

    /// Sets whether to include failed transactions
    pub fn include_failed_transactions(mut self, include: bool) -> Self {
        self.filter.include_failed = include;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ListenBotConfig::default();
        assert!(config.filter.program_ids.is_empty());
        assert!(config.filter.token_mints.is_empty());
        assert_eq!(config.filter.min_size, 0);
        assert!(!config.filter.include_failed);
        assert_eq!(config.rpc_url, "https://api.mainnet-beta.solana.com");
        assert_eq!(config.ws_url, "wss://api.mainnet-beta.solana.com");
        assert_eq!(config.max_concurrent_requests, 10);
        assert_eq!(config.request_timeout, 30);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_backoff_ms, 1000);
    }
} 