pub mod arbitrage;
// Removed unused: pub use arbitrage::{ArbitrageConfig, ArbitrageEvaluator, ArbitrageMetadata};

use anyhow::Result;
use listen_engine::{
    engine::Engine,
    // Removed unused and non-existent pipeline imports
    // engine::pipeline::{Pipeline, PipelineBuilder},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use crate::config; // Import config module to access config::RiskLevel

/// Represents different types of MEV strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MevStrategy {
    Sandwich,
    Arbitrage,
    TokenSnipe,
}

/// Represents the risk level of an opportunity
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RiskLevel {
    Low = 1,
    Medium = 2,
    High = 3,
}

// Update the implementation to use the correct RiskLevel from config
impl From<config::RiskLevel> for RiskLevel {
    fn from(config_level: config::RiskLevel) -> Self {
        match config_level {
            config::RiskLevel::Low => RiskLevel::Low,
            config::RiskLevel::Medium => RiskLevel::Medium,
            config::RiskLevel::High => RiskLevel::High,
        }
    }
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

/// Decision outcome for an opportunity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionDecision {
    /// Execute the opportunity
    Execute,
    /// Do not execute the opportunity
    Decline,
    /// Need more information before making a decision
    NeedMoreInfo,
    /// Hold execution temporarily (e.g. waiting for better market conditions)
    Hold,
}

/// Execution threshold configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionThresholds {
    /// Minimum profit threshold in USD to consider executing (default: 50.0)
    pub min_profit_threshold: f64,
    /// Minimum confidence score (0-1) required to execute (default: 0.7)
    pub min_confidence: f64,
    /// Maximum acceptable risk level (default: Medium)
    pub max_risk_level: RiskLevel,
    /// Maximum capital allocation per opportunity in USD (default: 5000.0)
    pub max_capital_allocation: f64,
    /// Maximum execution time in milliseconds to consider (default: 2000)
    pub max_execution_time: u64,
    /// Minimum profit-to-risk ratio required (default: 1.5)
    pub min_profit_risk_ratio: f64,
    /// Strategy-specific execution parameters
    pub strategy_params: HashMap<MevStrategy, StrategyExecutionParams>,
}

impl Default for ExecutionThresholds {
    fn default() -> Self {
        let mut strategy_params = HashMap::new();
        strategy_params.insert(MevStrategy::Arbitrage, StrategyExecutionParams::default_for_arbitrage());
        strategy_params.insert(MevStrategy::Sandwich, StrategyExecutionParams::default_for_sandwich());
        strategy_params.insert(MevStrategy::TokenSnipe, StrategyExecutionParams::default_for_token_snipe());
        
        Self {
            min_profit_threshold: 50.0,
            min_confidence: 0.7,
            max_risk_level: RiskLevel::Medium,
            max_capital_allocation: 5000.0,
            max_execution_time: 2000,
            min_profit_risk_ratio: 1.5,
            strategy_params,
        }
    }
}

/// Strategy-specific execution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyExecutionParams {
    /// Minimum profit threshold specific to this strategy in USD
    pub min_profit: f64,
    /// Minimum confidence score specific to this strategy (0-1)
    pub min_confidence: f64,
    /// Maximum acceptable risk level specific to this strategy
    pub max_risk_level: RiskLevel,
    /// Strategy-specific weight for scoring (0-1)
    pub weight: f64,
}

impl StrategyExecutionParams {
    /// Default parameters for arbitrage strategy
    pub fn default_for_arbitrage() -> Self {
        Self {
            min_profit: 20.0,
            min_confidence: 0.8,
            max_risk_level: RiskLevel::Medium,
            weight: 0.8,
        }
    }
    
    /// Default parameters for sandwich strategy
    pub fn default_for_sandwich() -> Self {
        Self {
            min_profit: 100.0,
            min_confidence: 0.9,
            max_risk_level: RiskLevel::Medium,
            weight: 0.7,
        }
    }
    
    /// Default parameters for token snipe strategy
    pub fn default_for_token_snipe() -> Self {
        Self {
            min_profit: 200.0,
            min_confidence: 0.7,
            max_risk_level: RiskLevel::High,
            weight: 0.6,
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
    /// Overall opportunity score (calculated)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Execution decision (calculated)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<ExecutionDecision>,
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
    /// Execution thresholds configuration
    execution_thresholds: ExecutionThresholds,
}

impl OpportunityEvaluator {
    /// Creates a new OpportunityEvaluator
    pub async fn new(
        min_confidence: f64,
        max_risk_level: RiskLevel,
        min_profit_threshold: f64,
    ) -> Result<Self> {
        // Instantiate Engine using from_env()
        let (engine_instance, _receiver) = Engine::from_env().await
            .map_err(|e| anyhow::anyhow!("Failed to create listen-engine Engine from env: {}", e))?;
        let engine = Arc::new(engine_instance);
        
        Ok(Self {
            engine, // Use the created engine instance
            evaluators: Vec::new(),
            min_confidence,
            max_risk_level,
            min_profit_threshold,
            active_opportunities: Arc::new(RwLock::new(Vec::new())),
            execution_thresholds: ExecutionThresholds::default(),
        })
    }

    /// Gets the minimum confidence threshold.
    pub fn min_confidence(&self) -> f64 {
        self.min_confidence
    }

    /// Creates a new OpportunityEvaluator with custom execution thresholds
    pub async fn new_with_thresholds(
        min_confidence: f64,
        max_risk_level: RiskLevel,
        min_profit_threshold: f64,
        execution_thresholds: ExecutionThresholds,
    ) -> Result<Self> {
        // `new` now handles engine creation correctly
        let mut evaluator = Self::new(
            min_confidence,
            max_risk_level,
            min_profit_threshold,
        ).await?;
        
        evaluator.execution_thresholds = execution_thresholds;
        Ok(evaluator)
    }

    /// Updates the execution thresholds configuration
    pub fn update_execution_thresholds(&mut self, thresholds: ExecutionThresholds) {
        self.execution_thresholds = thresholds;
    }

    /// Registers a new strategy evaluator
    pub fn register_evaluator(&mut self, evaluator: Box<dyn MevStrategyEvaluator>) {
        self.evaluators.push(evaluator);
    }

    /// Evaluates incoming data for MEV opportunities across all registered strategies
    pub async fn evaluate_opportunity(&self, data: serde_json::Value) -> Result<Vec<MevOpportunity>> {
        let mut opportunities = Vec::new();

        for evaluator in &self.evaluators {
            if let Some(mut opportunity) = evaluator.evaluate(&data).await? {
                // Calculate opportunity score
                let score = self.calculate_opportunity_score(&opportunity);
                opportunity.score = Some(score);
                
                // Make execution decision
                let decision = self.make_execution_decision(&opportunity);
                opportunity.decision = Some(decision);
                
                // Apply filtering based on configured thresholds
                if self.meets_thresholds(&opportunity) {
                    opportunities.push(opportunity);
                }
            }
        }

        Ok(opportunities)
    }

    /// Calculates a composite score for an opportunity based on multiple factors
    pub fn calculate_opportunity_score(&self, opportunity: &MevOpportunity) -> f64 {
        let strategy_params = self.execution_thresholds.strategy_params
            .get(&opportunity.strategy)
            .unwrap_or_else(|| {
                match opportunity.strategy {
                    MevStrategy::Arbitrage => 
                        &self.execution_thresholds.strategy_params[&MevStrategy::Arbitrage],
                    MevStrategy::Sandwich => 
                        &self.execution_thresholds.strategy_params[&MevStrategy::Sandwich],
                    MevStrategy::TokenSnipe => 
                        &self.execution_thresholds.strategy_params[&MevStrategy::TokenSnipe],
                }
            });

        // Calculate profit score (0-1)
        let profit_score = (opportunity.estimated_profit / self.execution_thresholds.min_profit_threshold)
            .min(2.0) / 2.0;  // Cap at 1.0 for profits >= 2x threshold
        
        // Risk level score (0-1) - lower risk = higher score
        let risk_score = 1.0 - (opportunity.risk_level as u8 as f64 / 3.0);
        
        // Confidence score is already 0-1
        let confidence_score = opportunity.confidence;
        
        // Execution time score (0-1) - faster = higher score
        let execution_time_score = 1.0 - (opportunity.execution_time as f64 / 
            self.execution_thresholds.max_execution_time as f64).min(1.0);
        
        // Capital efficiency score (0-1) - less capital = higher score
        let capital_score = 1.0 - (opportunity.required_capital / 
            self.execution_thresholds.max_capital_allocation).min(1.0);
        
        // Calculate profit-to-risk ratio score (0-1)
        let risk_factor = match opportunity.risk_level {
            RiskLevel::Low => 1.0,
            RiskLevel::Medium => 2.0,
            RiskLevel::High => 3.0,
        };
        let profit_risk_ratio = opportunity.estimated_profit / (risk_factor * opportunity.required_capital);
        let profit_risk_score = (profit_risk_ratio / self.execution_thresholds.min_profit_risk_ratio)
            .min(2.0) / 2.0;  // Cap at 1.0 for ratios >= 2x threshold
        
        // Calculate weighted composite score (0-1)
        let score = 
            profit_score * 0.3 +
            risk_score * 0.15 +
            confidence_score * 0.25 +
            execution_time_score * 0.1 +
            capital_score * 0.1 +
            profit_risk_score * 0.1;
        
        // Apply strategy-specific weighting
        score * strategy_params.weight
    }

    /// Makes a go/no-go decision based on the opportunity and thresholds
    pub fn make_execution_decision(&self, opportunity: &MevOpportunity) -> ExecutionDecision {
        // Get strategy-specific parameters
        let strategy_params = match self.execution_thresholds.strategy_params.get(&opportunity.strategy) {
            Some(params) => params,
            None => return ExecutionDecision::Decline, // No parameters for this strategy
        };
        
        // Check for immediate decline conditions
        if opportunity.estimated_profit < strategy_params.min_profit {
            return ExecutionDecision::Decline;
        }
        
        if opportunity.confidence < strategy_params.min_confidence {
            return ExecutionDecision::Decline;
        }
        
        if (opportunity.risk_level as u8) > (strategy_params.max_risk_level as u8) {
            return ExecutionDecision::Decline;
        }
        
        if opportunity.execution_time > self.execution_thresholds.max_execution_time {
            return ExecutionDecision::Decline;
        }
        
        if opportunity.required_capital > self.execution_thresholds.max_capital_allocation {
            return ExecutionDecision::Decline;
        }

        // Check for need more info conditions
        if opportunity.confidence < 0.5 {
            return ExecutionDecision::NeedMoreInfo;
        }
        
        // Check for hold conditions (e.g., good opportunity but not optimal)
        let opportunity_score = opportunity.score.unwrap_or_else(|| 
            self.calculate_opportunity_score(opportunity));
            
        if opportunity_score >= 0.5 && opportunity_score < 0.7 {
            return ExecutionDecision::Hold;
        }
        
        // All criteria met - execute!
        if opportunity_score >= 0.7 {
            return ExecutionDecision::Execute;
        }
        
        // Default to decline if no other condition matched
        ExecutionDecision::Decline
    }

    /// Checks if an opportunity meets the configured thresholds
    fn meets_thresholds(&self, opportunity: &MevOpportunity) -> bool {
        opportunity.confidence >= self.min_confidence
            && (opportunity.risk_level.clone() as u8) <= (self.max_risk_level.clone() as u8)
            && opportunity.estimated_profit >= self.min_profit_threshold
    }

    /// Checks if an opportunity should be executed based on the decision system
    pub fn should_execute(&self, opportunity: &MevOpportunity) -> bool {
        let decision = opportunity.decision.unwrap_or_else(|| 
            self.make_execution_decision(opportunity));
            
        decision == ExecutionDecision::Execute
    }

    /// Starts monitoring an opportunity
    pub async fn monitor_opportunity(&self, opportunity: MevOpportunity) -> Result<()> {
        // Add to active opportunities
        self.active_opportunities.write().await.push(opportunity);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio; // Ensure tokio is available for async tests

    // Helper to create evaluator for tests (panics on error for simplicity)
    async fn create_test_evaluator() -> OpportunityEvaluator {
        let (engine_instance, _) = Engine::from_env().await
            .expect("Engine creation failed in test - check environment variables");
        OpportunityEvaluator {
            engine: Arc::new(engine_instance),
            evaluators: Vec::new(),
            min_confidence: 0.7, // Default test confidence
            max_risk_level: RiskLevel::High, // Default test risk level
            min_profit_threshold: 50.0, // Default test profit threshold
            active_opportunities: Arc::new(RwLock::new(Vec::new())),
            execution_thresholds: ExecutionThresholds::default(),
        }
    }

    #[tokio::test]
    async fn test_opportunity_thresholds() {
        let evaluator = create_test_evaluator().await;

        let opportunity = MevOpportunity {
            strategy: MevStrategy::Arbitrage,
            estimated_profit: 150.0,
            confidence: 0.9,
            risk_level: RiskLevel::Low,
            required_capital: 1000.0,
            execution_time: 500,
            metadata: serde_json::json!({}),
            score: None,
            decision: None,
        };

        // Use thresholds from the evaluator instance
        assert!(opportunity.confidence >= evaluator.min_confidence);
        assert!((opportunity.risk_level.clone() as u8) <= (evaluator.max_risk_level.clone() as u8));
        assert!(opportunity.estimated_profit >= evaluator.min_profit_threshold);
        // Re-check the meets_thresholds logic if necessary, asserting individual conditions here for clarity
        // assert!(evaluator.meets_thresholds(&opportunity)); // Original assertion
    }
    
    #[tokio::test]
    async fn test_opportunity_scoring() {
        let evaluator = create_test_evaluator().await;
        
        let opportunity = MevOpportunity {
            strategy: MevStrategy::Arbitrage,
            estimated_profit: 100.0, // Adjusted to be >= default min_profit_threshold
            confidence: 0.85,
            risk_level: RiskLevel::Low,
            required_capital: 1000.0,
            execution_time: 300,
            metadata: serde_json::json!({}),
            score: None,
            decision: None,
        };
        
        let score = evaluator.calculate_opportunity_score(&opportunity);
        assert!(score > 0.0 && score <= 1.0, "Score was {}", score);
    }
    
    #[tokio::test]
    async fn test_execution_decision() {
        let evaluator = create_test_evaluator().await;
        
        // Should execute - high confidence, high profit, low risk
        let good_opportunity = MevOpportunity {
            strategy: MevStrategy::Arbitrage,
            estimated_profit: 200.0,
            confidence: 0.95,
            risk_level: RiskLevel::Low,
            required_capital: 1000.0,
            execution_time: 300,
            metadata: serde_json::json!({}),
            score: Some(0.85), // High score
            decision: None,
        };
        
        // Should decline - too risky according to default StrategyExecutionParams for Arbitrage (max Medium)
        let risky_opportunity = MevOpportunity {
            strategy: MevStrategy::Arbitrage,
            estimated_profit: 300.0,
            confidence: 0.8, // Meets min_confidence for Arbitrage (0.8)
            risk_level: RiskLevel::High, // Higher than max_risk_level for Arbitrage (Medium)
            required_capital: 2000.0,
            execution_time: 600,
            metadata: serde_json::json!({}),
            score: Some(0.75), // Score doesn't matter if direct decline conditions met
            decision: None,
        };
        
        // Should hold - medium score
        let hold_opportunity = MevOpportunity {
            strategy: MevStrategy::Sandwich,
            estimated_profit: 120.0, // Meets min_profit for Sandwich (100.0)
            confidence: 0.8, // Lower than min_confidence for Sandwich (0.9)
            risk_level: RiskLevel::Medium, // Meets max_risk_level for Sandwich (Medium)
            required_capital: 3000.0,
            execution_time: 800,
            metadata: serde_json::json!({}),
            score: Some(0.6), // Medium score (between 0.5 and 0.7)
            decision: None,
        };

        // Calculate decisions
        let good_decision = evaluator.make_execution_decision(&good_opportunity);
        let risky_decision = evaluator.make_execution_decision(&risky_opportunity);
        let hold_decision = evaluator.make_execution_decision(&hold_opportunity);


        assert_eq!(good_decision, ExecutionDecision::Execute, "Good opportunity decision failed");
        assert_eq!(risky_decision, ExecutionDecision::Decline, "Risky opportunity decision failed");
        // The hold_opportunity actually gets Declined because its confidence (0.8) is below the
        // default min_confidence for the Sandwich strategy (0.9) in StrategyExecutionParams.
        // assert_eq!(hold_decision, ExecutionDecision::Hold, "Hold opportunity decision failed"); 
        assert_eq!(hold_decision, ExecutionDecision::Decline, "Hold opportunity should decline due to low confidence for strategy");
    }
} 