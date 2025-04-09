mod arbitrage;
pub use arbitrage::{ArbitrageConfig, ArbitrageEvaluator, ArbitrageMetadata};

use anyhow::Result;
use listen_engine::{
    engine::{Engine, EngineConfig},
    pipeline::{Pipeline, PipelineBuilder},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Represents different types of MEV strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MevStrategy {
    Sandwich,
    Arbitrage,
    TokenSnipe,
}

/// Represents the risk level of an opportunity
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RiskLevel {
    Low = 1,
    Medium = 2,
    High = 3,
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

/// Core structure for MEV opportunities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MevOpportunity {
    /// Type of MEV strategy
    pub strategy: MevStrategy,
    /// Estimated profit in USD
    pub estimated_profit: f64,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Risk assessment
    pub risk_level: RiskLevel,
    /// Required capital in USD
    pub required_capital: f64,
    /// Estimated execution time in milliseconds
    pub execution_time: u64,
    /// Additional metadata for the opportunity
    pub metadata: serde_json::Value,
}

/// Interface for MEV strategy evaluators
#[async_trait::async_trait]
pub trait MevStrategyEvaluator: Send + Sync {
    /// Returns the type of MEV strategy this evaluator handles
    fn strategy_type(&self) -> MevStrategy;
    
    /// Evaluates a potential MEV opportunity
    async fn evaluate(&self, data: &serde_json::Value) -> Result<Option<MevOpportunity>>;
    
    /// Validates if an opportunity is still valid
    async fn validate(&self, opportunity: &MevOpportunity) -> Result<bool>;
}

/// Main opportunity evaluator that coordinates different strategy evaluators
pub struct OpportunityEvaluator {
    /// The underlying listen-engine instance
    engine: Arc<Engine>,
    /// Strategy evaluators
    evaluators: Vec<Box<dyn MevStrategyEvaluator>>,
    /// Minimum confidence threshold for opportunities
    min_confidence: f64,
    /// Maximum risk level allowed
    max_risk_level: RiskLevel,
    /// Minimum profit threshold in USD
    min_profit_threshold: f64,
    /// Active opportunities being tracked
    active_opportunities: Arc<RwLock<Vec<MevOpportunity>>>,
}

impl OpportunityEvaluator {
    /// Creates a new OpportunityEvaluator
    pub async fn new(
        engine_config: EngineConfig,
        min_confidence: f64,
        max_risk_level: RiskLevel,
        min_profit_threshold: f64,
    ) -> Result<Self> {
        let engine = Arc::new(Engine::new(engine_config).await?);
        
        Ok(Self {
            engine,
            evaluators: Vec::new(),
            min_confidence,
            max_risk_level,
            min_profit_threshold,
            active_opportunities: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Registers a new strategy evaluator
    pub fn register_evaluator(&mut self, evaluator: Box<dyn MevStrategyEvaluator>) {
        self.evaluators.push(evaluator);
    }

    /// Evaluates incoming data for MEV opportunities across all registered strategies
    pub async fn evaluate_opportunity(&self, data: serde_json::Value) -> Result<Vec<MevOpportunity>> {
        let mut opportunities = Vec::new();

        for evaluator in &self.evaluators {
            if let Some(opportunity) = evaluator.evaluate(&data).await? {
                // Apply filtering based on configured thresholds
                if self.meets_thresholds(&opportunity) {
                    opportunities.push(opportunity);
                }
            }
        }

        Ok(opportunities)
    }

    /// Checks if an opportunity meets the configured thresholds
    fn meets_thresholds(&self, opportunity: &MevOpportunity) -> bool {
        opportunity.confidence >= self.min_confidence
            && (opportunity.risk_level.clone() as u8) <= (self.max_risk_level.clone() as u8)
            && opportunity.estimated_profit >= self.min_profit_threshold
    }

    /// Creates a pipeline for monitoring an opportunity
    pub async fn create_opportunity_pipeline(&self, opportunity: &MevOpportunity) -> Result<Pipeline> {
        let mut builder = PipelineBuilder::new();
        
        // Add basic monitoring conditions based on strategy type
        match opportunity.strategy {
            MevStrategy::Sandwich => {
                // TODO: Add sandwich-specific monitoring conditions
            },
            MevStrategy::Arbitrage => {
                // TODO: Add arbitrage-specific monitoring conditions
            },
            MevStrategy::TokenSnipe => {
                // TODO: Add token snipe-specific monitoring conditions
            },
        }

        Ok(builder.build())
    }

    /// Starts monitoring an opportunity
    pub async fn monitor_opportunity(&self, opportunity: MevOpportunity) -> Result<()> {
        let pipeline = self.create_opportunity_pipeline(&opportunity).await?;
        self.engine.add_pipeline(pipeline).await?;
        
        // Add to active opportunities
        self.active_opportunities.write().await.push(opportunity);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_opportunity_thresholds() {
        let config = EngineConfig::new()
            .with_rpc_url("http://localhost:8899")
            .with_ws_url("ws://localhost:8900")
            .with_redis_url("redis://localhost:6379");

        let evaluator = OpportunityEvaluator::new(
            config,
            0.8, // min confidence
            RiskLevel::Medium,
            100.0, // min profit threshold
        ).await.unwrap();

        let opportunity = MevOpportunity {
            strategy: MevStrategy::Arbitrage,
            estimated_profit: 150.0,
            confidence: 0.9,
            risk_level: RiskLevel::Low,
            required_capital: 1000.0,
            execution_time: 500,
            metadata: serde_json::json!({}),
        };

        assert!(evaluator.meets_thresholds(&opportunity));
    }
} 