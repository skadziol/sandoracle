use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

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

#[derive(Debug)]
pub struct MarketDataCollector {
    // Price cache from the engine
    price_cache: Arc<RwLock<HashMap<String, f64>>>,
    // Historical price data from ClickHouse
    historical_data: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    // Market metrics cache
    market_metrics: Arc<RwLock<HashMap<String, MarketData>>>,
}

impl MarketDataCollector {
    pub fn new(
        price_cache: Arc<RwLock<HashMap<String, f64>>>,
        historical_data: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    ) -> Self {
        Self {
            price_cache,
            historical_data,
            market_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn get_market_data(&self, asset: &str) -> Result<Option<MarketData>> {
        // First check cache
        if let Some(data) = self.market_metrics.read().await.get(asset).cloned() {
            return Ok(Some(data));
        }

        // Get current price
        let price = match self.price_cache.read().await.get(asset).cloned() {
            Some(p) => p,
            None => return Ok(None),
        };

        // Get historical data
        let history = self.historical_data.read().await.get(asset).cloned();
        
        // Calculate metrics
        let (volume_24h, price_change_24h, volatility) = if let Some(hist) = history {
            let len = hist.len();
            if len >= 2 {
                let latest = hist[len - 1];
                let prev_24h = hist[len.saturating_sub(24)];
                let price_change = ((latest - prev_24h) / prev_24h) * 100.0;
                
                // Simple volatility calculation
                let volatility = hist.windows(2)
                    .map(|w| ((w[1] - w[0]) / w[0]).abs())
                    .sum::<f64>() / (len - 1) as f64;
                
                // Estimate volume from price changes
                let volume = hist.windows(2)
                    .map(|w| (w[1] - w[0]).abs() * latest)
                    .sum::<f64>();
                
                (volume, price_change, volatility)
            } else {
                (0.0, 0.0, 0.0)
            }
        } else {
            (0.0, 0.0, 0.0)
        };

        let market_data = MarketData {
            price,
            market_cap: price * 1_000_000.0, // Placeholder, needs token supply
            volume_24h,
            price_change_24h,
            liquidity: volume_24h * 0.1, // Estimate liquidity as 10% of volume
            volatility,
            last_update: chrono::Utc::now().timestamp() as u64,
        };

        // Update cache
        self.market_metrics.write().await.insert(asset.to_string(), market_data.clone());

        Ok(Some(market_data))
    }

    pub async fn get_all_market_data(&self) -> Result<HashMap<String, MarketData>> {
        let mut result = HashMap::new();
        let assets: Vec<String> = self.price_cache.read().await.keys().cloned().collect();
        
        for asset in assets {
            if let Ok(Some(data)) = self.get_market_data(&asset).await {
                result.insert(asset, data);
            }
        }
        
        Ok(result)
    }

    pub async fn update_market_data(&self, asset: &str, data: MarketData) -> Result<()> {
        self.market_metrics.write().await.insert(asset.to_string(), data);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokio::sync::RwLock;

    async fn setup_test_data() -> MarketDataCollector {
        let mut price_data = HashMap::new();
        price_data.insert("SOL".to_string(), 100.0);
        price_data.insert("BTC".to_string(), 50000.0);
        
        let mut historical_data = HashMap::new();
        historical_data.insert("SOL".to_string(), vec![95.0, 98.0, 100.0]); // 5.26% increase
        historical_data.insert("BTC".to_string(), vec![48000.0, 49000.0, 50000.0]); // 4.17% increase

        let price_cache = Arc::new(RwLock::new(price_data));
        let historical_data = Arc::new(RwLock::new(historical_data));

        MarketDataCollector::new(price_cache, historical_data)
    }

    #[tokio::test]
    async fn test_get_market_data() {
        let collector = setup_test_data().await;

        // Test SOL market data
        let sol_data = collector.get_market_data("SOL").await.unwrap().unwrap();
        assert_eq!(sol_data.price, 100.0);
        assert!(sol_data.price_change_24h > 5.0); // Should be around 5.26%
        assert!(sol_data.volatility > 0.0);
        assert!(sol_data.volume_24h > 0.0);

        // Test BTC market data
        let btc_data = collector.get_market_data("BTC").await.unwrap().unwrap();
        assert_eq!(btc_data.price, 50000.0);
        assert!(btc_data.price_change_24h > 4.0); // Should be around 4.17%
        assert!(btc_data.volatility > 0.0);
        assert!(btc_data.volume_24h > 0.0);

        // Test non-existent asset
        let none_data = collector.get_market_data("NONE").await.unwrap();
        assert!(none_data.is_none());
    }

    #[tokio::test]
    async fn test_get_all_market_data() {
        let collector = setup_test_data().await;
        
        let all_data = collector.get_all_market_data().await.unwrap();
        assert_eq!(all_data.len(), 2);
        assert!(all_data.contains_key("SOL"));
        assert!(all_data.contains_key("BTC"));
        
        // Verify data consistency
        let sol_data = all_data.get("SOL").unwrap();
        assert_eq!(sol_data.price, 100.0);
        
        let btc_data = all_data.get("BTC").unwrap();
        assert_eq!(btc_data.price, 50000.0);
    }

    #[tokio::test]
    async fn test_update_market_data() {
        let collector = setup_test_data().await;
        
        let new_data = MarketData {
            price: 110.0,
            market_cap: 11_000_000.0,
            volume_24h: 1_000_000.0,
            price_change_24h: 10.0,
            liquidity: 100_000.0,
            volatility: 0.05,
            last_update: chrono::Utc::now().timestamp() as u64,
        };
        
        // Update SOL data
        collector.update_market_data("SOL", new_data.clone()).await.unwrap();
        
        // Verify update
        let updated_data = collector.get_market_data("SOL").await.unwrap().unwrap();
        assert_eq!(updated_data.price, 110.0);
        assert_eq!(updated_data.market_cap, 11_000_000.0);
        assert_eq!(updated_data.volume_24h, 1_000_000.0);
        assert_eq!(updated_data.price_change_24h, 10.0);
        assert_eq!(updated_data.liquidity, 100_000.0);
        assert_eq!(updated_data.volatility, 0.05);
    }
} 