use serde::Deserialize;
use config::{Config, ConfigBuilder, ConfigError, Environment, File};
use config::builder::DefaultState;
use std::path::Path;
use std::env;
use std::convert::TryFrom;
use std::str::FromStr;

#[derive(Debug, Deserialize)]
pub struct Settings {
    // Solana configuration
    pub solana_rpc_url: String,
    pub wallet_private_key: String,

    // API keys
    pub rig_api_key: String,
    pub telegram_bot_token: Option<String>,

    // Performance settings
    pub max_concurrent_trades: usize,
    pub min_profit_threshold: f64,
    pub max_slippage: f64,

    // Monitoring configuration
    pub telegram_chat_id: Option<String>,
    pub log_level: String,

    // DEX API endpoints
    pub orca_api_url: String,
    pub raydium_api_url: String,
    pub jupiter_api_url: String,

    // Risk management
    pub max_position_size: f64,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

impl FromStr for RiskLevel {
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

impl TryFrom<Config> for Settings {
    type Error = ConfigError;

    fn try_from(config: Config) -> Result<Self, Self::Error> {
        Ok(Settings {
            solana_rpc_url: config.get_string("solana_rpc_url")?,
            wallet_private_key: config.get_string("wallet_private_key")?,
            rig_api_key: config.get_string("rig_api_key")?,
            telegram_bot_token: config.get_string("telegram_bot_token").ok(),
            max_concurrent_trades: config.get_int("max_concurrent_trades")? as usize,
            min_profit_threshold: config.get_float("min_profit_threshold")?,
            max_slippage: config.get_float("max_slippage")?,
            telegram_chat_id: config.get_string("telegram_chat_id").ok(),
            log_level: config.get_string("log_level")?,
            orca_api_url: config.get_string("orca_api_url")?,
            raydium_api_url: config.get_string("raydium_api_url")?,
            jupiter_api_url: config.get_string("jupiter_api_url")?,
            max_position_size: config.get_float("max_position_size")?,
            risk_level: config.get_string("risk_level")?.parse()?,
        })
    }
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        let env = env::var("RUN_MODE").unwrap_or_else(|_| "development".into());

        let config = ConfigBuilder::<DefaultState>::default()
            .add_source(File::with_name(&format!("config/{}", env)).required(false))
            .add_source(File::with_name("config/local").required(false))
            .add_source(Environment::with_prefix("APP"))
            .build()?;

        Settings::try_from(config)
    }

    pub fn from_env() -> Result<Self, ConfigError> {
        let config = ConfigBuilder::<DefaultState>::default()
            .add_source(Environment::default())
            .build()?;

        Settings::try_from(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_from_env() {
        std::env::set_var("SOLANA_RPC_URL", "https://test.solana.com");
        std::env::set_var("WALLET_PRIVATE_KEY", "test_key");
        std::env::set_var("RIG_API_KEY", "test_rig_key");
        std::env::set_var("MAX_CONCURRENT_TRADES", "5");
        std::env::set_var("MIN_PROFIT_THRESHOLD", "0.01");
        std::env::set_var("MAX_SLIPPAGE", "0.02");
        std::env::set_var("LOG_LEVEL", "info");
        std::env::set_var("ORCA_API_URL", "https://test.orca.so");
        std::env::set_var("RAYDIUM_API_URL", "https://test.raydium.io");
        std::env::set_var("JUPITER_API_URL", "https://test.jup.ag");
        std::env::set_var("MAX_POSITION_SIZE", "1000");
        std::env::set_var("RISK_LEVEL", "medium");

        let settings = Settings::from_env().unwrap();
        assert_eq!(settings.solana_rpc_url, "https://test.solana.com");
        assert_eq!(settings.max_concurrent_trades, 5);
        assert_eq!(settings.min_profit_threshold, 0.01);
    }
} 