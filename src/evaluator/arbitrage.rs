use crate::evaluator::{MevStrategy, MevOpportunity, MevStrategyEvaluator, RiskLevel};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use tracing::{debug, info, trace, warn};
use std::collections::HashMap;
use anyhow::anyhow;

/// Configuration for arbitrage opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageConfig {
    /// Minimum percentage profit required (e.g., 0.005 = 0.5%)
    pub min_profit_percentage: f64,
    /// Minimum USD value for trade to be considered
    pub min_trade_value_usd: f64,
    /// Maximum price impact allowed for arbitrage trades
    pub max_price_impact_percent: f64,
    /// Maximum time window for arbitrage to be valid (milliseconds)
    pub max_time_window_ms: u64,
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            min_profit_percentage: 0.005, // 0.5%
            min_trade_value_usd: 100.0,  // $100 minimum
            max_price_impact_percent: 1.0, // 1% max price impact
            max_time_window_ms: 2000,    // 2 second window
        }
    }
}

/// Metadata for arbitrage opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageMetadata {
    /// DEX where the first trade occurs
    pub source_dex: String,
    /// DEX where the second trade occurs
    pub target_dex: String,
    /// Token path for the arbitrage (A -> B -> A)
    pub token_path: Vec<String>,
    /// Estimated price difference (percentage)
    pub price_difference_percent: f64,
    /// Estimated gas costs in USD
    pub estimated_gas_cost_usd: f64,
    /// Optimal trade size in USD
    pub optimal_trade_size_usd: f64,
    /// Price impact of the trade
    pub price_impact_percent: f64,
}

/// Evaluator for arbitrage opportunities
pub struct ArbitrageEvaluator {
    config: ArbitrageConfig,
}

impl ArbitrageEvaluator {
    /// Create a new arbitrage evaluator with default configuration
    pub fn new() -> Self {
        Self {
            config: ArbitrageConfig::default(),
        }
    }
    
    /// Create a new arbitrage evaluator with custom configuration
    pub fn with_config(config: ArbitrageConfig) -> Self {
        Self { config }
    }
    
    /// Extract token pairs from transaction data
    fn extract_token_pairs(&self, data: &Value) -> Option<(String, String)> {
        // Extract transaction logs and look for token transfers
        if let Some(transaction) = data.get("transaction") {
            if let Some(_logs) = transaction.get("logs") {
                // Look for token transfers in logs
                trace!(target: "arbitrage_evaluator", "Analyzing logs for token transfers");
                
                // Very basic extraction - in a real implementation this would parse
                // the logs to find actual token transfers and addresses
                
                // For now just return a placeholder
                return Some(("SOL".to_string(), "USDC".to_string()));
            }
        }
        None
    }
    
    /// Calculate the potential profit from an arbitrage opportunity
    fn calculate_profit(&self, source_price: f64, target_price: f64, trade_size_usd: f64) -> f64 {
        let profit_percentage = (target_price - source_price) / source_price;
        trade_size_usd * profit_percentage - self.config.min_trade_value_usd * 0.001 // Subtract estimated fees
    }
    
    /// Determine risk level based on the arbitrage opportunity
    fn determine_risk_level(&self, price_diff_percent: f64, liquidity_usd: f64) -> RiskLevel {
        if price_diff_percent > 5.0 || liquidity_usd < 10000.0 {
            RiskLevel::High
        } else if price_diff_percent > 2.0 || liquidity_usd < 50000.0 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }
}

#[async_trait]
impl MevStrategyEvaluator for ArbitrageEvaluator {
    fn strategy_type(&self) -> MevStrategy {
        MevStrategy::Arbitrage
    }
    
    async fn evaluate(&self, data: &Value) -> Result<Option<MevOpportunity>> {
        trace!(target: "arbitrage_evaluator", "Evaluating potential arbitrage opportunity");
        // Use real DEX price data if available
        let price_map = data.get("real_dex_prices")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow!("No real_dex_prices in input data"))?;
        // Collect all DEX prices (filter out None)
        let mut dex_prices: Vec<(String, f64)> = price_map.iter()
            .filter_map(|(dex, price)| price.as_f64().map(|p| (dex.clone(), p)))
            .collect();
        if dex_prices.len() < 2 {
            trace!(target: "arbitrage_evaluator", "Not enough DEX prices for arbitrage");
            return Ok(None);
        }
        // Find best buy (lowest price) and best sell (highest price)
        dex_prices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let (buy_dex, buy_price) = &dex_prices[0];
        let (sell_dex, sell_price) = &dex_prices[dex_prices.len() - 1];
        let price_difference_percent = (sell_price - buy_price) / buy_price * 100.0;
        // Only consider if spread is above config threshold
        if price_difference_percent < self.config.min_profit_percentage * 100.0 {
            trace!(target: "arbitrage_evaluator", "No profitable arbitrage spread");
            return Ok(None);
        }
        // Estimate profit (for demo, use $1000 trade size)
        let trade_size_usd = 1000.0;
        let estimated_profit_usd = (sell_price - buy_price) / buy_price * trade_size_usd;
        let liquidity_usd = 50000.0; // TODO: Use real liquidity if available
        let risk_level = self.determine_risk_level(price_difference_percent, liquidity_usd);
        let required_capital = trade_size_usd;
        let arb_metadata = ArbitrageMetadata {
            source_dex: buy_dex.clone(),
            target_dex: sell_dex.clone(),
            token_path: vec!["SOL".to_string(), "USDC".to_string(), "SOL".to_string()],
            price_difference_percent,
            estimated_gas_cost_usd: 0.2, // TODO: Use real gas estimate
            optimal_trade_size_usd: trade_size_usd,
            price_impact_percent: 0.1, // TODO: Use real price impact
        };
        let metadata = serde_json::to_value(arb_metadata)?;
        let opportunity = MevOpportunity {
            strategy: MevStrategy::Arbitrage,
            estimated_profit: estimated_profit_usd,
            confidence: 0.9, // TODO: Compute based on data quality
            risk_level,
            required_capital,
            execution_time: 800, // ms, TODO: Estimate
            metadata,
            score: None,
            decision: None,
            involved_tokens: vec!["SOL".to_string(), "USDC".to_string()],
            allowed_output_tokens: vec!["SOL".to_string(), "USDC".to_string()],
            allowed_programs: vec![buy_dex.clone(), sell_dex.clone()],
            max_instructions: 10,
        };
        info!(
            target: "arbitrage_evaluator",
            profit = estimated_profit_usd,
            price_diff = price_difference_percent,
            buy_dex = %buy_dex,
            sell_dex = %sell_dex,
            "Found real arbitrage opportunity"
        );
        Ok(Some(opportunity))
    }
    
    async fn validate(&self, _opportunity: &MevOpportunity) -> Result<bool> {
        // In a real implementation, you would re-check prices to ensure the opportunity still exists
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