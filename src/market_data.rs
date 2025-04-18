use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::error::{Result, SandoError};
use crate::jupiter_client::Jupiter;
use tracing::{info, error};

/// Collector for market data from various sources
#[derive(Clone)]
pub struct MarketDataCollector {
    price_cache: Arc<RwLock<HashMap<String, f64>>>,
    historical_data: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    jupiter_client: Option<Jupiter>,
}

/// Market data for a specific token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub price: f64,
    pub market_cap: f64,
    pub volume_24h: f64,
    pub price_change_24h: f64,
    pub liquidity: f64,
    pub volatility: f64,
    pub last_update: u64,
}

impl MarketDataCollector {
    /// Create a new market data collector
    pub fn new(
        price_cache: Arc<RwLock<HashMap<String, f64>>>,
        historical_data: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    ) -> Self {
        Self {
            price_cache,
            historical_data,
            jupiter_client: None,
        }
    }

    /// Initialize with Jupiter API for real-time price data
    pub fn with_jupiter(mut self, jupiter_api_url: &str) -> Result<Self> {
        self.jupiter_client = Some(Jupiter::new(jupiter_api_url));
        info!("Initialized Jupiter client for real-time market data");
        Ok(self)
    }

    /// Update market data for a token
    pub async fn update_market_data(&self, token: &str, data: MarketData) -> Result<()> {
        // Update price in cache
        let mut cache = self.price_cache.write().await;
        cache.insert(token.to_string(), data.price);
        
        // Update historical data (keep last 24 data points)
        let mut history = self.historical_data.write().await;
        let token_history = history.entry(token.to_string()).or_insert_with(Vec::new);
        token_history.push(data.price);
        
        // Keep only the last 24 data points
        if token_history.len() > 24 {
            token_history.remove(0);
        }
        
        Ok(())
    }

    /// Fetch real market data from Jupiter API
    pub async fn fetch_real_market_data(&self, token: &str, reference_token: &str) -> Result<Option<f64>> {
        // If token and reference token are the same, the price is always 1:1
        if token == reference_token {
            info!(token=%token, "Token and reference token are identical, returning 1.0 as price");
            return Ok(Some(1.0));
        }
        
        if let Some(jupiter) = &self.jupiter_client {
            match jupiter.get_price(token, reference_token).await {
                Ok(price) => return Ok(Some(price)),
                Err(e) => {
                    error!(token=%token, error=%e, "Failed to fetch price from Jupiter");
                    return Ok(None);
                }
            }
        }
        
        // If no Jupiter client or on error, try to get from cache
        let cache = self.price_cache.read().await;
        Ok(cache.get(token).copied())
    }
}