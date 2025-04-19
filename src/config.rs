use serde::{Deserialize, Serialize};
// Remove unused imports
// use std::collections::HashMap;
// use std::time::Duration;
use config::ConfigError;
use std::str::FromStr;
use std::env;

#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    // Solana Configuration
    pub solana_rpc_url: String,
    pub solana_ws_url: String, // Added
    pub wallet_private_key: String,

    // RIG Agent Configuration
    pub rig_api_key: Option<String>,
    pub rig_model_provider: Option<String>, // Changed from ModelProvider to String
    pub rig_model_name: Option<String>,

    // Monitoring & Notifications
    pub telegram_bot_token: Option<String>,
    pub telegram_chat_id: Option<String>,
    pub log_level: String,

    // Trading Parameters
    pub max_concurrent_trades: u32,
    pub min_profit_threshold: f64,
    pub max_slippage: f64,
    pub max_position_size: f64,
    pub risk_level: RiskLevel,
    pub simulation_mode: bool, // Added
    pub allowed_tokens: Option<Vec<String>>,
    pub blocked_tokens: Option<Vec<String>>,

    // Listen Bot Specific (Added)
    pub request_timeout_secs: Option<u64>,
    pub max_retries: Option<u32>,
    pub retry_backoff_ms: Option<u64>,

    // DEX Specific APIs (Optional)
    pub orca_api_url: Option<String>,
    pub raydium_api_url: Option<String>,
    pub jupiter_api_url: Option<String>,

    // Detailed MEV Thresholds (Optional - Consider a nested struct)
    pub arbitrage_min_profit: Option<f64>,
    pub sandwich_min_profit: Option<f64>,
    pub snipe_min_profit: Option<f64>,

    // Geyser Configuration
    pub geyser_plugin_endpoint_url: String,

    // Solana Configuration
    pub commitment: Option<String>, // Added commitment level

    /// Enable mempool monitoring (if false, uses confirmed blocks)
    pub use_mempool: Option<bool>,
}

impl Settings {
    pub fn from_env() -> Result<Self, ConfigError> {
        Ok(Settings {
            log_level: env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            solana_rpc_url: env::var("SOLANA_RPC_URL")
                .map_err(|_| ConfigError::Message("Missing SOLANA_RPC_URL env var".into()))?,
            solana_ws_url: env::var("SOLANA_WS_URL")
                .map_err(|_| ConfigError::Message("Missing SOLANA_WS_URL env var".into()))?,
            rig_api_key: env::var("RIG_API_KEY").ok(),
            rig_model_provider: env::var("RIG_MODEL_PROVIDER").ok(),
            rig_model_name: env::var("RIG_MODEL_NAME").ok(),
            telegram_bot_token: env::var("TELEGRAM_BOT_TOKEN").ok(),
            telegram_chat_id: env::var("TELEGRAM_CHAT_ID").ok(),
            min_profit_threshold: env::var("MIN_PROFIT_THRESHOLD")
                .unwrap_or_else(|_| "0.01".to_string())
                .parse::<f64>()
                .map_err(|_| ConfigError::Message("Invalid MIN_PROFIT_THRESHOLD".into()))?,
            max_slippage: env::var("MAX_SLIPPAGE")
                .unwrap_or_else(|_| "0.02".to_string())
                .parse::<f64>()
                .map_err(|_| ConfigError::Message("Invalid MAX_SLIPPAGE".into()))?,
            max_position_size: env::var("MAX_POSITION_SIZE")
                .unwrap_or_else(|_| "1000.0".to_string())
                .parse::<f64>()
                .map_err(|_| ConfigError::Message("Invalid MAX_POSITION_SIZE".into()))?,
            risk_level: RiskLevel::from_str(&env::var("RISK_LEVEL").unwrap_or_else(|_| "medium".to_string()))
                .map_err(|_| ConfigError::Message("Invalid RISK_LEVEL".into()))?,
            simulation_mode: env::var("SIMULATION_MODE")
                .unwrap_or_else(|_| "true".to_string())
                .parse::<bool>()
                .map_err(|_| ConfigError::Message("Invalid SIMULATION_MODE".into()))?,
            max_concurrent_trades: env::var("MAX_CONCURRENT_TRADES")
                .unwrap_or_else(|_| "5".to_string())
                .parse::<u32>()
                .map_err(|_| ConfigError::Message("Invalid MAX_CONCURRENT_TRADES".into()))?,
            request_timeout_secs: env::var("REQUEST_TIMEOUT_SECS").ok().and_then(|v| v.parse().ok()),
            max_retries: env::var("MAX_RETRIES").ok().and_then(|v| v.parse().ok()),
            retry_backoff_ms: env::var("RETRY_BACKOFF_MS").ok().and_then(|v| v.parse().ok()),
            orca_api_url: env::var("ORCA_API_URL").ok(),
            raydium_api_url: env::var("RAYDIUM_API_URL").ok(),
            jupiter_api_url: env::var("JUPITER_API_URL").ok(),
            arbitrage_min_profit: env::var("ARBITRAGE_MIN_PROFIT").ok().and_then(|v| v.parse().ok()),
            sandwich_min_profit: env::var("SANDWICH_MIN_PROFIT").ok().and_then(|v| v.parse().ok()),
            snipe_min_profit: env::var("SNIPE_MIN_PROFIT").ok().and_then(|v| v.parse().ok()),
            wallet_private_key: env::var("WALLET_PRIVATE_KEY")
                .map_err(|_| ConfigError::Message("Missing WALLET_PRIVATE_KEY env var".into()))?,
            geyser_plugin_endpoint_url: env::var("GEYSER_PLUGIN_ENDPOINT_URL")
                .map_err(|_| ConfigError::Message("Missing GEYSER_PLUGIN_ENDPOINT_URL env var".into()))?,
            commitment: env::var("SOLANA_COMMITMENT_LEVEL").ok(),
            allowed_tokens: env::var("ALLOWED_TOKENS").ok().map(|s| s.split(',').map(String::from).collect()),
            blocked_tokens: env::var("BLOCKED_TOKENS").ok().map(|s| s.split(',').map(String::from).collect()),
            use_mempool: env::var("USE_MEMPOOL").ok().and_then(|v| v.parse().ok()),
        })
    }

    #[cfg(test)]
    pub fn default_for_tests() -> Self {
        Self {
            solana_rpc_url: "https://api.testnet.solana.com".to_string(),
            solana_ws_url: "wss://api.testnet.solana.com".to_string(),
            wallet_private_key: "Test1111111111111111111111111111111111111111111".to_string(),
            simulation_mode: true,
            rig_api_key: Some("test_api_key".to_string()),
            rig_model_provider: Some("anthropic".to_string()),
            rig_model_name: Some("claude-3-opus".to_string()),
            telegram_bot_token: Some("test_bot_token".to_string()),
            telegram_chat_id: Some("test_chat_id".to_string()),
            log_level: "info".to_string(),
            risk_level: RiskLevel::Medium,
            max_concurrent_trades: 3,
            min_profit_threshold: 10.0,
            max_slippage: 0.01,
            max_position_size: 1000.0,
            allowed_tokens: Some(vec!["SOL".to_string(), "USDC".to_string(), "ETH".to_string()]),
            blocked_tokens: Some(vec![]),
            request_timeout_secs: Some(30),
            max_retries: Some(3),
            retry_backoff_ms: Some(500),
            orca_api_url: Some("https://api.testnet.orca.so".to_string()),
            raydium_api_url: Some("https://api.testnet.raydium.io".to_string()),
            jupiter_api_url: Some("https://quotes.jup.ag/v4".to_string()),
            arbitrage_min_profit: Some(10.0),
            sandwich_min_profit: Some(20.0),
            snipe_min_profit: Some(50.0),
            geyser_plugin_endpoint_url: "ws://localhost:8900/".to_string(),
            commitment: Some("processed".to_string()),
            use_mempool: Some(true),
        }
    }

    // Optional: Add getter methods if needed, e.g., for log level parsing
    // pub fn get_log_level(&self) -> tracing::Level {
    //     match self.log_level.as_deref() {
    //         Some("trace") => tracing::Level::TRACE,
    //         Some("debug") => tracing::Level::DEBUG,
    //         Some("info") => tracing::Level::INFO,
    //         Some("warn") => tracing::Level::WARN,
    //         Some("error") => tracing::Level::ERROR,
    //         _ => tracing::Level::INFO, // Default level
    //     }
    // }
}

/// Represents the risk level for MEV operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

impl std::str::FromStr for RiskLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(RiskLevel::Low),
            "medium" => Ok(RiskLevel::Medium),
            "high" => Ok(RiskLevel::High),
            _ => Err(format!("Invalid risk level: {}", s)),
        }
    }
} 