use serde::{Deserialize, Serialize};
// Remove unused imports
// use std::collections::HashMap;
// use std::time::Duration;

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
    pub log_level: Option<String>, // Consider using tracing_subscriber::filter::LevelFilter

    // Trading Parameters
    pub max_concurrent_trades: usize,
    pub min_profit_threshold: f64,
    pub max_slippage: f64,
    pub max_position_size: f64,
    pub risk_level: crate::evaluator::RiskLevel, // Use full path
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
}

impl Settings {
    pub fn from_env() -> Result<Self, config::ConfigError> {
        let config_builder = config::Config::builder()
            .add_source(config::Environment::default().separator("__"))
            .set_default("max_concurrent_trades", 5)?
            .set_default("min_profit_threshold", 0.01)?
            .set_default("max_slippage", 0.02)?
            .set_default("max_position_size", 1000.0)?
            .set_default("risk_level", "medium")? // Default as string, deserialized into enum
            .set_default("simulation_mode", true)? // Added default
            // Add defaults for listen bot specific settings
            .set_default("request_timeout_secs", 30)?
            .set_default("max_retries", 3)?
            .set_default("retry_backoff_ms", 1000)?;

        // Load optional .env file
        if dotenv::dotenv().is_ok() {
            println!("Loaded configuration from .env file");
        }

        let settings = config_builder.build()?;
        settings.try_deserialize()
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