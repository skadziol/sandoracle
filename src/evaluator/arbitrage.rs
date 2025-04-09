use super::{MevOpportunity, MevStrategy, MevStrategyEvaluator, RiskLevel};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for arbitrage evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageConfig {
    /// Minimum profit percentage required (e.g., 0.01 for 1%)
    pub min_profit_percentage: f64,
    /// Maximum price impact allowed (e.g., 0.02 for 2%)
    pub max_price_impact: f64,
    /// Minimum liquidity required in USD
    pub min_liquidity: f64,
    /// Maximum number of hops in the arbitrage path
    pub max_hops: u8,
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            min_profit_percentage: 0.01, // 1%
            max_price_impact: 0.02,      // 2%
            min_liquidity: 10000.0,      // $10k
            max_hops: 3,
        }
    }
}

/// Metadata specific to arbitrage opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageMetadata {
    /// The path of tokens in the arbitrage
    pub token_path: Vec<String>,
    /// Expected prices at each step
    pub prices: Vec<f64>,
    /// Liquidity available at each step
    pub liquidity: Vec<f64>,
    /// Estimated price impact at each step
    pub price_impacts: Vec<f64>,
    /// DEXes involved in the arbitrage
    pub dexes: Vec<String>,
}

pub struct ArbitrageEvaluator {
    config: ArbitrageConfig,
    /// Cache of token prices from different DEXes
    price_cache: HashMap<String, HashMap<String, f64>>,
}

impl ArbitrageEvaluator {
    pub fn new(config: ArbitrageConfig) -> Self {
        Self {
            config,
            price_cache: HashMap::new(),
        }
    }

    /// Updates the price cache with new price data
    pub fn update_prices(&mut self, token: String, dex: String, price: f64) {
        self.price_cache
            .entry(token)
            .or_default()
            .insert(dex, price);
    }

    /// Calculates the potential profit for an arbitrage path
    fn calculate_profit(
        &self,
        _token_path: &[String],
        amounts: &[f64],
        price_impacts: &[f64],
    ) -> f64 {
        let initial_amount = amounts[0];
        let final_amount = amounts.last().unwrap();
        
        // Calculate profit after considering price impact
        let profit = final_amount - initial_amount;
        let total_price_impact: f64 = price_impacts.iter().sum();
        
        profit * (1.0 - total_price_impact)
    }

    /// Validates liquidity is sufficient across the path
    fn validate_liquidity(&self, liquidity: &[f64]) -> bool {
        liquidity.iter().all(|&l| l >= self.config.min_liquidity)
    }

    /// Calculates confidence score based on various factors
    fn calculate_confidence(
        &self,
        profit_percentage: f64,
        price_impacts: &[f64],
        liquidity: &[f64],
    ) -> f64 {
        let profit_score = (profit_percentage / self.config.min_profit_percentage).min(1.0);
        let impact_score = price_impacts.iter().all(|&i| i <= self.config.max_price_impact) as u8 as f64;
        let liquidity_score = self.validate_liquidity(liquidity) as u8 as f64;
        
        // Weight the factors (can be adjusted based on historical performance)
        let weights = [0.4, 0.3, 0.3];
        profit_score * weights[0] + impact_score * weights[1] + liquidity_score * weights[2]
    }
}

#[async_trait::async_trait]
impl MevStrategyEvaluator for ArbitrageEvaluator {
    fn strategy_type(&self) -> MevStrategy {
        MevStrategy::Arbitrage
    }

    async fn evaluate(&self, data: &serde_json::Value) -> Result<Option<MevOpportunity>> {
        // Extract relevant data from the input
        let token_path: Vec<String> = serde_json::from_value(data["token_path"].clone())?;
        let amounts: Vec<f64> = serde_json::from_value(data["amounts"].clone())?;
        let amounts_clone = amounts.clone();
        let price_impacts: Vec<f64> = serde_json::from_value(data["price_impacts"].clone())?;
        let liquidity: Vec<f64> = serde_json::from_value(data["liquidity"].clone())?;
        let dexes: Vec<String> = serde_json::from_value(data["dexes"].clone())?;

        // Validate path length
        if token_path.len() > self.config.max_hops as usize + 1 {
            return Ok(None);
        }

        // Calculate profit and validate thresholds
        let profit = self.calculate_profit(&token_path, &amounts, &price_impacts);
        let profit_percentage = profit / amounts[0];
        
        if profit_percentage < self.config.min_profit_percentage {
            return Ok(None);
        }

        // Calculate confidence score
        let confidence = self.calculate_confidence(profit_percentage, &price_impacts, &liquidity);

        // Create metadata
        let metadata = ArbitrageMetadata {
            token_path: token_path.clone(),
            prices: amounts,
            liquidity,
            price_impacts,
            dexes,
        };

        let opportunity = MevOpportunity {
            strategy: MevStrategy::Arbitrage,
            estimated_profit: profit,
            confidence,
            risk_level: if profit_percentage > 0.05 { RiskLevel::High } else { RiskLevel::Medium },
            required_capital: amounts_clone[0],
            execution_time: (token_path.len() as u64) * 500, // Rough estimate: 500ms per hop
            metadata: serde_json::to_value(metadata)?,
        };

        Ok(Some(opportunity))
    }

    async fn validate(&self, opportunity: &MevOpportunity) -> Result<bool> {
        let metadata: ArbitrageMetadata = serde_json::from_value(opportunity.metadata.clone())?;
        
        // Revalidate liquidity
        if !self.validate_liquidity(&metadata.liquidity) {
            return Ok(false);
        }

        // Check if prices have moved significantly
        for (token, price) in metadata.token_path.iter().zip(metadata.prices.iter()) {
            if let Some(current_prices) = self.price_cache.get(token) {
                let avg_current_price: f64 = current_prices.values().sum::<f64>() / current_prices.len() as f64;
                let price_diff = (avg_current_price - price).abs() / price;
                
                if price_diff > 0.01 { // 1% price movement threshold
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_arbitrage_evaluation() {
        let config = ArbitrageConfig::default();
        let evaluator = ArbitrageEvaluator::new(config);

        let test_data = serde_json::json!({
            "token_path": ["USDC", "SOL", "USDC"],
            "amounts": [1000.0, 40.0, 1015.0],
            "price_impacts": [0.001, 0.001, 0.001],
            "liquidity": [100000.0, 50000.0, 100000.0],
            "dexes": ["Orca", "Raydium", "Orca"]
        });

        let result = evaluator.evaluate(&test_data).await.unwrap();
        assert!(result.is_some());

        let opportunity = result.unwrap();
        assert_eq!(opportunity.strategy, MevStrategy::Arbitrage);
        assert!(opportunity.estimated_profit > 0.0);
        assert!(opportunity.confidence > 0.5);
    }
} 