use super::{MevOpportunity, MevStrategy, MevStrategyEvaluator, RiskLevel};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for sandwich evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandwichConfig {
    /// Minimum profit percentage required (e.g., 0.02 for 2%)
    pub min_profit_percentage: f64,
    /// Maximum price impact allowed for our trades (e.g., 0.03 for 3%)
    pub max_price_impact: f64,
    /// Minimum target transaction size in USD
    pub min_target_tx_size: f64,
    /// Maximum target transaction size in USD
    pub max_target_tx_size: f64,
    /// Minimum liquidity required in USD
    pub min_liquidity: f64,
    /// Maximum time window for execution (ms)
    pub max_execution_time: u64,
}

impl Default for SandwichConfig {
    fn default() -> Self {
        Self {
            min_profit_percentage: 0.02, // 2%
            max_price_impact: 0.03,      // 3%
            min_target_tx_size: 5000.0,  // $5k
            max_target_tx_size: 100000.0, // $100k
            min_liquidity: 50000.0,      // $50k
            max_execution_time: 1000,     // 1 second
        }
    }
}

/// Metadata specific to sandwich opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandwichMetadata {
    /// Target transaction hash
    pub target_tx_hash: String,
    /// Token pair being traded
    pub token_pair: (String, String),
    /// Target transaction size in USD
    pub target_tx_size: f64,
    /// Front-run amount in USD
    pub front_run_amount: f64,
    /// Back-run amount in USD
    pub back_run_amount: f64,
    /// Expected price impact of front-run
    pub front_run_impact: f64,
    /// Expected price impact of back-run
    pub back_run_impact: f64,
    /// DEX being used
    pub dex: String,
    /// Estimated gas costs
    pub gas_cost: f64,
}

pub struct SandwichEvaluator {
    config: SandwichConfig,
    /// Cache of pool states
    pool_states: HashMap<String, PoolState>,
}

#[derive(Clone)]
struct PoolState {
    liquidity: f64,
    price: f64,
    last_update: std::time::SystemTime,
}

impl SandwichEvaluator {
    pub fn new(config: SandwichConfig) -> Self {
        Self {
            config,
            pool_states: HashMap::new(),
        }
    }

    /// Updates the pool state cache
    pub fn update_pool_state(&mut self, pool_id: String, liquidity: f64, price: f64) {
        self.pool_states.insert(pool_id, PoolState {
            liquidity,
            price,
            last_update: std::time::SystemTime::now(),
        });
    }

    /// Calculates the optimal front-run amount
    fn calculate_optimal_amounts(
        &self,
        target_tx_size: f64,
        pool_liquidity: f64,
    ) -> (f64, f64) {
        // Simple heuristic: front-run with 20-30% of target tx size
        let front_run_amount = target_tx_size * 0.25;
        let back_run_amount = front_run_amount * 1.05; // Slightly higher to account for slippage
        
        // Adjust based on pool liquidity
        let liquidity_ratio = front_run_amount / pool_liquidity;
        if liquidity_ratio > 0.1 {
            // Scale down if using too much liquidity
            let scale = 0.1 / liquidity_ratio;
            (front_run_amount * scale, back_run_amount * scale)
        } else {
            (front_run_amount, back_run_amount)
        }
    }

    /// Calculates expected profit after gas costs
    fn calculate_net_profit(
        &self,
        front_run_amount: f64,
        back_run_amount: f64,
        price_impact: f64,
        gas_cost: f64,
    ) -> f64 {
        let gross_profit = back_run_amount - front_run_amount;
        gross_profit * (1.0 - price_impact) - gas_cost
    }

    /// Calculates confidence score based on various factors
    fn calculate_confidence(
        &self,
        target_tx_size: f64,
        pool_liquidity: f64,
        price_impact: f64,
        execution_time: u64,
    ) -> f64 {
        let size_score = if target_tx_size >= self.config.min_target_tx_size 
            && target_tx_size <= self.config.max_target_tx_size {
            1.0
        } else {
            0.0
        };

        let liquidity_score = if pool_liquidity >= self.config.min_liquidity {
            1.0
        } else {
            pool_liquidity / self.config.min_liquidity
        };

        let impact_score = if price_impact <= self.config.max_price_impact {
            1.0
        } else {
            0.0
        };

        let time_score = if execution_time <= self.config.max_execution_time {
            1.0
        } else {
            0.0
        };

        // Weight the factors
        let weights = [0.3, 0.3, 0.2, 0.2];
        size_score * weights[0] +
        liquidity_score * weights[1] +
        impact_score * weights[2] +
        time_score * weights[3]
    }
}

#[async_trait::async_trait]
impl MevStrategyEvaluator for SandwichEvaluator {
    fn strategy_type(&self) -> MevStrategy {
        MevStrategy::Sandwich
    }

    async fn evaluate(&self, data: &serde_json::Value) -> Result<Option<MevOpportunity>> {
        // Extract relevant data from the input
        let target_tx_hash: String = serde_json::from_value(data["tx_hash"].clone())?;
        let token_pair: (String, String) = serde_json::from_value(data["token_pair"].clone())?;
        let target_tx_size: f64 = serde_json::from_value(data["tx_size"].clone())?;
        let pool_id: String = serde_json::from_value(data["pool_id"].clone())?;
        let dex: String = serde_json::from_value(data["dex"].clone())?;
        let gas_cost: f64 = serde_json::from_value(data["gas_cost"].clone())?;

        // Get pool state
        let pool_state = if let Some(state) = self.pool_states.get(&pool_id) {
            state
        } else {
            return Ok(None); // No pool state available
        };

        // Validate transaction size
        if target_tx_size < self.config.min_target_tx_size 
            || target_tx_size > self.config.max_target_tx_size {
            return Ok(None);
        }

        // Calculate optimal amounts
        let (front_run_amount, back_run_amount) = 
            self.calculate_optimal_amounts(target_tx_size, pool_state.liquidity);

        // Estimate price impacts
        let front_run_impact = front_run_amount / pool_state.liquidity;
        let back_run_impact = back_run_amount / pool_state.liquidity;

        // Calculate profit
        let total_price_impact = front_run_impact + back_run_impact;
        let net_profit = self.calculate_net_profit(
            front_run_amount,
            back_run_amount,
            total_price_impact,
            gas_cost,
        );

        let profit_percentage = net_profit / front_run_amount;
        if profit_percentage < self.config.min_profit_percentage {
            return Ok(None);
        }

        // Calculate confidence
        let confidence = self.calculate_confidence(
            target_tx_size,
            pool_state.liquidity,
            total_price_impact,
            self.config.max_execution_time,
        );

        // Create metadata
        let metadata = SandwichMetadata {
            target_tx_hash,
            token_pair,
            target_tx_size,
            front_run_amount,
            back_run_amount,
            front_run_impact,
            back_run_impact,
            dex,
            gas_cost,
        };

        let opportunity = MevOpportunity {
            strategy: MevStrategy::Sandwich,
            estimated_profit: net_profit,
            confidence,
            risk_level: RiskLevel::High, // Sandwich attacks are inherently high risk
            required_capital: front_run_amount,
            execution_time: self.config.max_execution_time,
            metadata: serde_json::to_value(metadata)?,
            score: None,
            decision: None,
        };

        Ok(Some(opportunity))
    }

    async fn validate(&self, opportunity: &MevOpportunity) -> Result<bool> {
        let metadata: SandwichMetadata = serde_json::from_value(opportunity.metadata.clone())?;
        
        // Get pool state
        let pool_id = format!("{}_{}", metadata.dex, metadata.token_pair.0);
        let pool_state = if let Some(state) = self.pool_states.get(&pool_id) {
            state
        } else {
            return Ok(false);
        };

        // Check if pool state is fresh (within last 2 seconds)
        let age = pool_state.last_update.elapsed()?.as_millis();
        if age > 2000 {
            return Ok(false);
        }

        // Validate liquidity is still sufficient
        if pool_state.liquidity < self.config.min_liquidity {
            return Ok(false);
        }

        // Validate price impact is still within limits
        let total_impact = metadata.front_run_impact + metadata.back_run_impact;
        if total_impact > self.config.max_price_impact {
            return Ok(false);
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sandwich_evaluation() {
        let config = SandwichConfig::default();
        let mut evaluator = SandwichEvaluator::new(config);

        // Setup test pool state
        evaluator.update_pool_state(
            "orca_usdc_sol".to_string(),
            100000.0,
            20.0,
        );

        let test_data = serde_json::json!({
            "tx_hash": "0x1234567890abcdef",
            "token_pair": ["USDC", "SOL"],
            "tx_size": 10000.0,
            "pool_id": "orca_usdc_sol",
            "dex": "Orca",
            "gas_cost": 10.0
        });

        let result = evaluator.evaluate(&test_data).await.unwrap();
        assert!(result.is_some());

        let opportunity = result.unwrap();
        assert_eq!(opportunity.strategy, MevStrategy::Sandwich);
        assert!(opportunity.estimated_profit > 0.0);
        assert!(opportunity.confidence > 0.0);
        assert_eq!(opportunity.risk_level, RiskLevel::High);
    }
} 