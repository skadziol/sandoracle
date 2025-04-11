use super::{MevOpportunity, MevStrategy, MevStrategyEvaluator, RiskLevel};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, Duration};

/// Configuration for token snipe evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSnipeConfig {
    /// Minimum expected return multiplier (e.g., 2.0 for 2x)
    pub min_return_multiplier: f64,
    /// Maximum price impact allowed (e.g., 0.05 for 5%)
    pub max_price_impact: f64,
    /// Minimum initial liquidity required in USD
    pub min_initial_liquidity: f64,
    /// Maximum capital to deploy per opportunity
    pub max_capital: f64,
    /// Maximum time to wait for execution (ms)
    pub max_execution_time: u64,
    /// Minimum token age (seconds) to consider
    pub min_token_age: u64,
}

impl Default for TokenSnipeConfig {
    fn default() -> Self {
        Self {
            min_return_multiplier: 2.0,  // 2x return
            max_price_impact: 0.05,      // 5%
            min_initial_liquidity: 20000.0, // $20k
            max_capital: 2000.0,         // $2k
            max_execution_time: 500,      // 500ms
            min_token_age: 300,          // 5 minutes
        }
    }
}

/// Metadata specific to token snipe opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenSnipeMetadata {
    /// Token address
    pub token_address: String,
    /// Token name/symbol
    pub token_symbol: String,
    /// Initial price in USD
    pub initial_price: f64,
    /// Initial liquidity in USD
    pub initial_liquidity: f64,
    /// Token creation timestamp
    pub creation_time: i64,
    /// DEX where token is listed
    pub dex: String,
    /// Token contract verification status
    pub is_verified: bool,
    /// Token holder distribution metrics
    pub holder_metrics: TokenHolderMetrics,
    /// Estimated gas costs for execution
    pub gas_cost: f64,
}

/// Metrics about token holders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenHolderMetrics {
    /// Number of unique holders
    pub unique_holders: u32,
    /// Percentage of supply held by top holder
    pub top_holder_percentage: f64,
    /// Percentage of supply in liquidity pool
    pub liquidity_percentage: f64,
}

pub struct TokenSnipeEvaluator {
    config: TokenSnipeConfig,
    /// Cache of token states
    token_states: HashMap<String, TokenState>,
}

#[derive(Clone)]
struct TokenState {
    price: f64,
    liquidity: f64,
    holder_count: u32,
    last_update: SystemTime,
}

impl TokenSnipeEvaluator {
    pub fn new(config: TokenSnipeConfig) -> Self {
        Self {
            config,
            token_states: HashMap::new(),
        }
    }

    /// Updates the token state cache
    pub fn update_token_state(
        &mut self,
        token_address: String,
        price: f64,
        liquidity: f64,
        holder_count: u32,
    ) {
        self.token_states.insert(token_address, TokenState {
            price,
            liquidity,
            holder_count,
            last_update: SystemTime::now(),
        });
    }

    /// Calculates the risk score based on token metrics
    fn calculate_risk_score(&self, metadata: &TokenSnipeMetadata) -> f64 {
        let holder_metrics = &metadata.holder_metrics;
        
        // Factors that reduce risk
        let verification_score = if metadata.is_verified { 1.0 } else { 0.0 };
        let holder_distribution_score = 1.0 - holder_metrics.top_holder_percentage;
        let liquidity_score = (holder_metrics.liquidity_percentage / 0.5).min(1.0);
        let holder_count_score = (holder_metrics.unique_holders as f64 / 100.0).min(1.0);
        
        // Weight and combine factors
        let weights = [0.3, 0.3, 0.2, 0.2];
        verification_score * weights[0] +
        holder_distribution_score * weights[1] +
        liquidity_score * weights[2] +
        holder_count_score * weights[3]
    }

    /// Calculates confidence score based on various factors
    fn calculate_confidence(
        &self,
        metadata: &TokenSnipeMetadata,
        risk_score: f64,
    ) -> f64 {
        // Basic checks
        if metadata.initial_liquidity < self.config.min_initial_liquidity {
            return 0.0;
        }

        let token_age = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64 - metadata.creation_time;
            
        if token_age < self.config.min_token_age as i64 {
            return 0.0;
        }

        // Factor scores
        let liquidity_score = (metadata.initial_liquidity / self.config.min_initial_liquidity)
            .min(1.0);
        
        let age_score = (token_age as f64 / (self.config.min_token_age as f64 * 2.0))
            .min(1.0);
            
        let holder_score = metadata.holder_metrics.unique_holders as f64 / 100.0;

        // Combine scores with weights
        let weights = [0.4, 0.2, 0.2, 0.2];
        risk_score * weights[0] +
        liquidity_score * weights[1] +
        age_score * weights[2] +
        holder_score * weights[3]
    }
}

#[async_trait::async_trait]
impl MevStrategyEvaluator for TokenSnipeEvaluator {
    fn strategy_type(&self) -> MevStrategy {
        MevStrategy::TokenSnipe
    }

    async fn evaluate(&self, data: &serde_json::Value) -> Result<Option<MevOpportunity>> {
        // Extract token metadata
        let metadata: TokenSnipeMetadata = serde_json::from_value(data.clone())?;

        // Get current token state
        let token_state = if let Some(state) = self.token_states.get(&metadata.token_address) {
            state
        } else {
            return Ok(None); // No token state available
        };

        // Calculate risk score
        let risk_score = self.calculate_risk_score(&metadata);

        // Calculate confidence
        let confidence = self.calculate_confidence(&metadata, risk_score);
        if confidence == 0.0 {
            return Ok(None);
        }

        // Calculate potential return
        let price_ratio = token_state.price / metadata.initial_price;
        if price_ratio < self.config.min_return_multiplier {
            return Ok(None);
        }

        // Calculate required capital and estimated profit
        let required_capital = self.config.max_capital;
        let estimated_profit = required_capital * (price_ratio - 1.0) - metadata.gas_cost;

        // Determine risk level based on metrics
        let risk_level = if risk_score < 0.3 {
            RiskLevel::High
        } else if risk_score < 0.7 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        let opportunity = MevOpportunity {
            strategy: MevStrategy::TokenSnipe,
            estimated_profit,
            confidence,
            risk_level,
            required_capital,
            execution_time: self.config.max_execution_time,
            metadata: serde_json::to_value(metadata)?,
            score: None,
            decision: None,
        };

        Ok(Some(opportunity))
    }

    async fn validate(&self, opportunity: &MevOpportunity) -> Result<bool> {
        let metadata: TokenSnipeMetadata = serde_json::from_value(opportunity.metadata.clone())?;
        
        // Get current token state
        let token_state = if let Some(state) = self.token_states.get(&metadata.token_address) {
            state
        } else {
            return Ok(false);
        };

        // Check if state is fresh (within last 5 seconds)
        let age = token_state.last_update.elapsed()?.as_millis();
        if age > 5000 {
            return Ok(false);
        }

        // Validate liquidity is still sufficient
        if token_state.liquidity < self.config.min_initial_liquidity {
            return Ok(false);
        }

        // Validate price hasn't moved unfavorably
        let current_price_ratio = token_state.price / metadata.initial_price;
        if current_price_ratio < self.config.min_return_multiplier {
            return Ok(false);
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_snipe_evaluation() {
        let config = TokenSnipeConfig::default();
        let mut evaluator = TokenSnipeEvaluator::new(config);

        // Setup test token state
        evaluator.update_token_state(
            "0xtoken123".to_string(),
            0.02, // Current price
            50000.0, // Liquidity
            150, // Holder count
        );

        let test_data = serde_json::json!({
            "token_address": "0xtoken123",
            "token_symbol": "TEST",
            "initial_price": 0.01,
            "initial_liquidity": 25000.0,
            "creation_time": SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as i64 - 600,
            "dex": "Orca",
            "is_verified": true,
            "holder_metrics": {
                "unique_holders": 150,
                "top_holder_percentage": 0.2,
                "liquidity_percentage": 0.4
            },
            "gas_cost": 5.0
        });

        let result = evaluator.evaluate(&test_data).await.unwrap();
        assert!(result.is_some());

        let opportunity = result.unwrap();
        assert_eq!(opportunity.strategy, MevStrategy::TokenSnipe);
        assert!(opportunity.estimated_profit > 0.0);
        assert!(opportunity.confidence > 0.0);
    }
} 