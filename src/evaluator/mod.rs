pub mod arbitrage;
pub mod sandwich;
pub mod token_snipe;

// Remove or uncomment these imports
use arbitrage::ArbitrageEvaluator;
use sandwich::SandwichEvaluator;
use token_snipe::TokenSnipeEvaluator;

// use listen_engine::Engine; // REMOVE THIS LINE

use crate::config::Settings;
use crate::error::{Result as SandoResult, SandoError};
use crate::config::RiskLevel as ConfigRiskLevel;
use crate::rig_agent::{RigAgent, RawOpportunityData, MarketContext};
use anyhow::{Result, Error, Context, anyhow};

// Removed unused: pub use arbitrage::{ArbitrageConfig, ArbitrageEvaluator, ArbitrageMetadata};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn, trace};
use crate::config; // Import config module to access config::RiskLevel
use std::time::{SystemTime, UNIX_EPOCH};
use chrono;
use crate::monitoring::{OPPORTUNITY_LOGGER, OPPORTUNITY_RATE_LOGGER};

/// Represents different types of MEV strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MevStrategy {
    Arbitrage,
    Sandwich,
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
    /// List of tokens involved in the transaction
    pub involved_tokens: Vec<String>,
    /// List of allowed output tokens (tokens that can decrease in balance)
    pub allowed_output_tokens: Vec<String>,
    /// List of allowed program IDs that can be called
    pub allowed_programs: Vec<String>,
    /// Maximum number of instructions allowed
    pub max_instructions: u64,
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

/// Registry for MEV strategy evaluators
#[derive(Default)]
pub struct StrategyRegistry {
    evaluators: HashMap<MevStrategy, Box<dyn MevStrategyEvaluator>>,
}

impl StrategyRegistry {
    pub fn new() -> Self {
        Self {
            evaluators: HashMap::new(),
        }
    }

    pub fn register(&mut self, evaluator: Box<dyn MevStrategyEvaluator>) -> Option<Box<dyn MevStrategyEvaluator>> {
        let strategy_type = evaluator.strategy_type();
        debug!(strategy = ?strategy_type, "Registering strategy evaluator");
        self.evaluators.insert(strategy_type, evaluator)
    }

    pub fn get(&self, strategy: &MevStrategy) -> Option<&dyn MevStrategyEvaluator> {
        self.evaluators.get(strategy).map(|e| e.as_ref())
    }

    pub fn contains_strategy(&self, strategy: &MevStrategy) -> bool {
        self.evaluators.contains_key(strategy)
    }

    pub fn remove(&mut self, strategy: &MevStrategy) -> Option<Box<dyn MevStrategyEvaluator>> {
        debug!(strategy = ?strategy, "Removing strategy evaluator");
        self.evaluators.remove(strategy)
    }

    pub fn strategies(&self) -> Vec<MevStrategy> {
        self.evaluators.keys().cloned().collect()
    }
}

/// Represents a scoring factor that contributes to the overall opportunity score
#[async_trait::async_trait]
pub trait ScoringFactor: Send + Sync {
    /// Returns the name of this scoring factor
    fn name(&self) -> &'static str;
    
    /// Returns the weight of this factor (0.0 to 1.0)
    fn weight(&self) -> f64;
    
    /// Calculate the score contribution for this factor
    async fn calculate_score(&self, opportunity: &MevOpportunity, thresholds: &ExecutionThresholds) -> f64;
    
    /// Returns true if this factor is critical (i.e., if it fails, the opportunity should be rejected)
    fn is_critical(&self) -> bool {
        false
    }
}

/// Profit-based scoring factor
pub struct ProfitFactor {
    weight: f64,
}

impl ProfitFactor {
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }
}

#[async_trait::async_trait]
impl ScoringFactor for ProfitFactor {
    fn name(&self) -> &'static str {
        "profit"
    }
    
    fn weight(&self) -> f64 {
        self.weight
    }
    
    async fn calculate_score(&self, opportunity: &MevOpportunity, thresholds: &ExecutionThresholds) -> f64 {
        let min_profit = thresholds.strategy_params
            .get(&opportunity.strategy)
            .map(|params| params.min_profit)
            .unwrap_or_else(|| {
                // If no params in thresholds, use default
                let default_params = OpportunityEvaluator::get_strategy_params(&opportunity.strategy);
                default_params.min_profit
            });
            
        if opportunity.estimated_profit < min_profit {
            return 0.0;
        }
        
        // Score based on how much the profit exceeds the minimum threshold
        let profit_ratio = opportunity.estimated_profit / min_profit;
        (1.0 - (1.0 / profit_ratio)).min(1.0)
    }
    
    fn is_critical(&self) -> bool {
        true
    }
}

/// Time sensitivity scoring factor
pub struct TimeSensitivityFactor {
    weight: f64,
}

impl TimeSensitivityFactor {
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }
}

#[async_trait::async_trait]
impl ScoringFactor for TimeSensitivityFactor {
    fn name(&self) -> &'static str {
        "time_sensitivity"
    }
    
    fn weight(&self) -> f64 {
        self.weight
    }
    
    async fn calculate_score(&self, opportunity: &MevOpportunity, thresholds: &ExecutionThresholds) -> f64 {
        let max_time = thresholds.max_execution_time as f64;
        let time = opportunity.execution_time as f64;
        
        if time > max_time {
            return 0.0;
        }
        
        // Score based on how quickly we can execute compared to max time
        1.0 - (time / max_time)
    }
}

/// Risk-adjusted return scoring factor
pub struct RiskAdjustedReturnFactor {
    weight: f64,
}

impl RiskAdjustedReturnFactor {
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }
}

#[async_trait::async_trait]
impl ScoringFactor for RiskAdjustedReturnFactor {
    fn name(&self) -> &'static str {
        "risk_adjusted_return"
    }
    
    fn weight(&self) -> f64 {
        self.weight
    }
    
    async fn calculate_score(&self, opportunity: &MevOpportunity, thresholds: &ExecutionThresholds) -> f64 {
        let risk_multiplier = match opportunity.risk_level {
            RiskLevel::Low => 1.0,
            RiskLevel::Medium => 0.7,
            RiskLevel::High => 0.4,
        };
        
        let profit_ratio = opportunity.estimated_profit / opportunity.required_capital;
        let risk_adjusted_ratio = profit_ratio * risk_multiplier;
        
        if risk_adjusted_ratio < thresholds.min_profit_risk_ratio {
            return 0.0;
        }
        
        (1.0 - (thresholds.min_profit_risk_ratio / risk_adjusted_ratio)).min(1.0)
    }
    
    fn is_critical(&self) -> bool {
        true
    }
}

/// Represents market conditions that might trigger circuit breakers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    /// Volatility index (0-100)
    pub volatility_index: f64,
    /// Recent price change percentage
    pub price_change_percent: f64,
    /// Trading volume in USD
    pub volume_usd: f64,
    /// Liquidity depth in USD
    pub liquidity_depth: f64,
    /// Number of similar transactions in mempool
    pub mempool_density: u32,
    /// Gas price in GWEI
    pub gas_price_gwei: u64,
    /// Timestamp of the conditions
    pub timestamp: i64,
}

/// Circuit breaker status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerStatus {
    /// Normal operation
    Normal,
    /// Warning level - increased scrutiny needed
    Warning,
    /// Critical level - halt operations
    Critical,
    /// Temporary pause
    Paused,
}

/// Circuit breaker for monitoring specific market conditions
#[async_trait::async_trait]
pub trait CircuitBreaker: Send + Sync {
    /// Returns the name of this circuit breaker
    fn name(&self) -> &'static str;
    
    /// Check if the circuit breaker should be triggered
    async fn check(&self, conditions: &MarketConditions) -> CircuitBreakerStatus;
    
    /// Returns true if this is a critical circuit breaker
    fn is_critical(&self) -> bool {
        false
    }
}

/// Monitors extreme volatility conditions
pub struct VolatilityBreaker {
    /// Volatility threshold for warning level
    warning_threshold: f64,
    /// Volatility threshold for critical level
    critical_threshold: f64,
}

impl VolatilityBreaker {
    pub fn default() -> Self {
        Self {
            warning_threshold: 25.0,  // 25% volatility
            critical_threshold: 40.0,  // 40% volatility
        }
    }
}

#[async_trait::async_trait]
impl CircuitBreaker for VolatilityBreaker {
    fn name(&self) -> &'static str {
        "volatility"
    }
    
    async fn check(&self, conditions: &MarketConditions) -> CircuitBreakerStatus {
        if conditions.volatility_index >= self.critical_threshold {
            warn!(
                volatility = conditions.volatility_index,
                threshold = self.critical_threshold,
                "Volatility circuit breaker triggered: Critical"
            );
            CircuitBreakerStatus::Critical
        } else if conditions.volatility_index >= self.warning_threshold {
            warn!(
                volatility = conditions.volatility_index,
                threshold = self.warning_threshold,
                "Volatility circuit breaker triggered: Warning"
            );
            CircuitBreakerStatus::Warning
        } else {
            CircuitBreakerStatus::Normal
        }
    }
    
    fn is_critical(&self) -> bool {
        true
    }
}

/// Monitors gas price spikes
pub struct GasPriceBreaker {
    /// Gas price threshold for warning level (in GWEI)
    warning_threshold: u64,
    /// Gas price threshold for critical level (in GWEI)
    critical_threshold: u64,
}

impl GasPriceBreaker {
    pub fn default() -> Self {
        Self {
            warning_threshold: 200,  // 200 GWEI
            critical_threshold: 500,  // 500 GWEI
        }
    }
}

#[async_trait::async_trait]
impl CircuitBreaker for GasPriceBreaker {
    fn name(&self) -> &'static str {
        "gas_price"
    }
    
    async fn check(&self, conditions: &MarketConditions) -> CircuitBreakerStatus {
        if conditions.gas_price_gwei >= self.critical_threshold {
            warn!(
                gas_price = conditions.gas_price_gwei,
                threshold = self.critical_threshold,
                "Gas price circuit breaker triggered: Critical"
            );
            CircuitBreakerStatus::Critical
        } else if conditions.gas_price_gwei >= self.warning_threshold {
            warn!(
                gas_price = conditions.gas_price_gwei,
                threshold = self.warning_threshold,
                "Gas price circuit breaker triggered: Warning"
            );
            CircuitBreakerStatus::Warning
        } else {
            CircuitBreakerStatus::Normal
        }
    }
}

/// Monitors mempool congestion
pub struct MempoolBreaker {
    /// Mempool density threshold for warning level
    warning_threshold: u32,
    /// Mempool density threshold for critical level
    critical_threshold: u32,
}

impl MempoolBreaker {
    pub fn default() -> Self {
        Self {
            warning_threshold: 50,   // 50 similar transactions
            critical_threshold: 100,  // 100 similar transactions
        }
    }
}

#[async_trait::async_trait]
impl CircuitBreaker for MempoolBreaker {
    fn name(&self) -> &'static str {
        "mempool"
    }
    
    async fn check(&self, conditions: &MarketConditions) -> CircuitBreakerStatus {
        if conditions.mempool_density >= self.critical_threshold {
            warn!(
                density = conditions.mempool_density,
                threshold = self.critical_threshold,
                "Mempool circuit breaker triggered: Critical"
            );
            CircuitBreakerStatus::Critical
        } else if conditions.mempool_density >= self.warning_threshold {
            warn!(
                density = conditions.mempool_density,
                threshold = self.warning_threshold,
                "Mempool circuit breaker triggered: Warning"
            );
            CircuitBreakerStatus::Warning
        } else {
            CircuitBreakerStatus::Normal
        }
    }
}

/// Monitors strategy-specific market conditions
pub struct StrategyBreaker {
    /// Thresholds for arbitrage opportunities
    arbitrage_thresholds: ArbitrageThresholds,
    /// Thresholds for sandwich opportunities
    sandwich_thresholds: SandwichThresholds,
    /// Thresholds for token snipe opportunities
    token_snipe_thresholds: TokenSnipeThresholds,
}

#[derive(Clone)]
struct ArbitrageThresholds {
    max_price_deviation: f64,
    min_liquidity_ratio: f64,
}

#[derive(Clone)]
struct SandwichThresholds {
    max_pending_txs: u32,
    min_target_tx_size: f64,
}

#[derive(Clone)]
struct TokenSnipeThresholds {
    min_holder_count: u32,
    min_liquidity_percentage: f64,
}

impl StrategyBreaker {
    pub fn default() -> Self {
        Self {
            arbitrage_thresholds: ArbitrageThresholds {
                max_price_deviation: 0.05, // 5%
                min_liquidity_ratio: 0.1,  // 10%
            },
            sandwich_thresholds: SandwichThresholds {
                max_pending_txs: 5,
                min_target_tx_size: 5000.0, // $5k
            },
            token_snipe_thresholds: TokenSnipeThresholds {
                min_holder_count: 50,
                min_liquidity_percentage: 0.2, // 20%
            },
        }
    }

    fn check_arbitrage(&self, conditions: &MarketConditions) -> CircuitBreakerStatus {
        // Check price deviation
        if conditions.price_change_percent.abs() > self.arbitrage_thresholds.max_price_deviation {
            return CircuitBreakerStatus::Warning;
        }

        // Check liquidity ratio
        let liquidity_ratio = conditions.liquidity_depth / conditions.volume_usd;
        if liquidity_ratio < self.arbitrage_thresholds.min_liquidity_ratio {
            return CircuitBreakerStatus::Warning;
        }

        CircuitBreakerStatus::Normal
    }

    fn check_sandwich(&self, conditions: &MarketConditions) -> CircuitBreakerStatus {
        // Check mempool density
        if conditions.mempool_density > self.sandwich_thresholds.max_pending_txs {
            return CircuitBreakerStatus::Warning;
        }

        // Check if gas price is too high for profitable sandwich
        if conditions.gas_price_gwei > 300 {
            return CircuitBreakerStatus::Critical;
        }

        CircuitBreakerStatus::Normal
    }

    fn check_token_snipe(&self, conditions: &MarketConditions) -> CircuitBreakerStatus {
        // Token snipe opportunities are more sensitive to market conditions
        if conditions.volatility_index > 30.0 {
            return CircuitBreakerStatus::Critical;
        }

        if conditions.price_change_percent.abs() > 0.1 {
            return CircuitBreakerStatus::Warning;
        }

        CircuitBreakerStatus::Normal
    }
}

#[async_trait::async_trait]
impl CircuitBreaker for StrategyBreaker {
    fn name(&self) -> &'static str {
        "strategy_breaker"
    }

    async fn check(&self, conditions: &MarketConditions) -> CircuitBreakerStatus {
        // Check all strategies and return the most severe status
        let statuses = vec![
            self.check_arbitrage(conditions),
            self.check_sandwich(conditions),
            self.check_token_snipe(conditions),
        ];

        if statuses.iter().any(|s| matches!(s, CircuitBreakerStatus::Critical)) {
            CircuitBreakerStatus::Critical
        } else if statuses.iter().any(|s| matches!(s, CircuitBreakerStatus::Warning)) {
            CircuitBreakerStatus::Warning
        } else {
            CircuitBreakerStatus::Normal
        }
    }

    fn is_critical(&self) -> bool {
        true
    }
}

/// Main opportunity evaluator that coordinates different strategy evaluators
pub struct OpportunityEvaluator {
    /// Strategy registry
    strategy_registry: StrategyRegistry,
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
    /// Strategy executor
    strategy_executor: RwLock<Option<Arc<dyn StrategyExecutionService>>>,
    /// RIG Agent for AI-powered decision enhancement
    rig_agent: Option<Arc<RigAgent>>,
}

/// This trait defines the interface for strategy execution services
#[async_trait::async_trait]
pub trait StrategyExecutionService: Send + Sync {
    /// Executes an MEV opportunity
    async fn execute_opportunity(&self, opportunity: &MevOpportunity) -> anyhow::Result<solana_sdk::signature::Signature>;
}

impl OpportunityEvaluator {
    /// Creates a new OpportunityEvaluator with default thresholds.
    pub async fn new() -> Result<Self> {
        let thresholds = ExecutionThresholds::default();
        Self::new_with_thresholds(
            thresholds.min_profit_threshold, 
            thresholds.max_risk_level, 
            0.05, // Default min_profit_percentage, adjust if needed
            thresholds
        ).await
    }

    /// Creates a new OpportunityEvaluator with specified thresholds.
    pub async fn new_with_thresholds(
        min_profit_threshold: f64,
        max_risk_level: RiskLevel,
        _min_profit_percentage: f64, // Parameter might be unused now or repurposed
        execution_thresholds: ExecutionThresholds,
    ) -> Result<Self> {
        info!("Initializing OpportunityEvaluator...");

        let registry = StrategyRegistry::new();
        // Add registration logic here if needed later

        // Removed engine initialization
        // let (engine, _price_rx) = Engine::from_env().await
        //     .map_err(|e| SandoError::DependencyError(format!("Failed to create listen-engine: {}", e)))?;

        let evaluator = Self {
            // engine: Arc::new(engine), // Removed engine field assignment
            strategy_registry: registry,
            min_confidence: execution_thresholds.min_confidence,
            max_risk_level,
            min_profit_threshold,
            active_opportunities: Arc::new(RwLock::new(Vec::new())),
            execution_thresholds,
            strategy_executor: RwLock::new(None), // Initialize executor as None
            rig_agent: None,
        };

        info!("OpportunityEvaluator initialized successfully.");
        Ok(evaluator)
    }

    /// Sets the strategy executor
    pub async fn set_strategy_executor<T: StrategyExecutionService + 'static>(&mut self, executor: Arc<T>) {
        let mut strategy_executor = self.strategy_executor.write().await;
        *strategy_executor = Some(executor);
    }

    /// Executes an MEV opportunity
    pub async fn execute_opportunity(&self, opportunity: &MevOpportunity) -> Result<solana_sdk::signature::Signature> {
        // Check if we have a strategy executor
        let strategy_executor = self.strategy_executor.read().await;
        
        if let Some(executor) = strategy_executor.as_ref() {
            // Execute the opportunity
            executor.execute_opportunity(opportunity).await
                .map_err(|e| anyhow::anyhow!("{}", e).into()) // Convert anyhow::Error to SandoError
        } else {
            Err(anyhow::anyhow!("No strategy executor available").into())
        }
    }

    /// Processes a detected MEV opportunity, evaluating and potentially executing it
    pub async fn process_mev_opportunity(&self, opportunity: &mut MevOpportunity) -> Result<Option<solana_sdk::signature::Signature>> {
        // First, calculate the opportunity score if not already done
        if opportunity.score.is_none() {
            let score = self.calculate_opportunity_score(opportunity).await;
            opportunity.score = Some(score);
        }
        
        // Then, make execution decision if not already done
        if opportunity.decision.is_none() {
            let decision = self.make_execution_decision(opportunity).await;
            opportunity.decision = Some(decision);
        }
        
        // Check if we should execute this opportunity
        if self.should_execute(opportunity).await {
            info!(
                strategy = ?opportunity.strategy,
                estimated_profit = opportunity.estimated_profit,
                confidence = opportunity.confidence,
                score = opportunity.score.unwrap_or(0.0),
                "Executing MEV opportunity"
            );
            
            // Execute the opportunity
            match self.execute_opportunity(opportunity).await {
                Ok(signature) => {
                    info!(
                        signature = %signature,
                        "Successfully executed MEV opportunity"
                    );
                    Ok(Some(signature))
                },
                Err(e) => {
                    error!(
                        error = %e,
                        "Failed to execute MEV opportunity"
                    );
                    Err(e)
                }
            }
        } else {
            info!(
                strategy = ?opportunity.strategy,
                decision = ?opportunity.decision,
                "Decided not to execute MEV opportunity"
            );
            Ok(None)
        }
    }

    /// Gets the minimum confidence threshold.
    pub fn min_confidence(&self) -> f64 {
        self.min_confidence
    }

    /// Updates the execution thresholds configuration
    pub fn update_execution_thresholds(&mut self, thresholds: ExecutionThresholds) {
        self.execution_thresholds = thresholds;
    }

    /// Registers a new strategy evaluator
    pub fn register_evaluator(&mut self, evaluator: Box<dyn MevStrategyEvaluator>) {
        let strategy_type = evaluator.strategy_type();
        info!(strategy = ?strategy_type, "Registering new strategy evaluator");
        if let Some(_) = self.strategy_registry.register(evaluator) {
            warn!(
                strategy = ?strategy_type,
                "Replaced existing evaluator for strategy"
            );
        }
    }

    /// Evaluates incoming data for MEV opportunities across all registered strategies
    pub async fn evaluate_opportunity(&self, data: serde_json::Value) -> Result<Vec<MevOpportunity>> {
        let should_log_detailed = OPPORTUNITY_RATE_LOGGER.should_log() || 
            std::env::var("DETAILED_LOGGING").map(|v| v == "true").unwrap_or(false);
            
        if should_log_detailed {
            debug!(target: "opportunity_evaluation", data_size = data.to_string().len(), "Starting opportunity evaluation");
        } else {
            trace!(target: "opportunity_evaluation", "Evaluating opportunity data");
        }
        
        let mut opportunities = Vec::new();
        for (strategy_name, evaluator) in &self.strategy_registry.evaluators {
            // Use trace for routine evaluations to reduce log volume
            trace!(target: "opportunity_evaluation::strategy", strategy = ?strategy_name, "Evaluating with strategy");
            
            match evaluator.evaluate(&data).await {
                Ok(Some(mut opportunity)) => {
                    // Calculate score for the opportunity
                    let score = self.calculate_opportunity_score(&opportunity).await;
                    opportunity.score = Some(score);
                    
                    // Make execution decision
                    let decision = self.make_execution_decision(&opportunity).await;
                    opportunity.decision = Some(decision);
                    
                    if should_log_detailed || decision == ExecutionDecision::Execute {
                        debug!(
                            target: "opportunity_evaluation::result",
                            strategy = ?strategy_name,
                            score,
                            decision = ?decision,
                            estimated_profit = opportunity.estimated_profit,
                            "Opportunity evaluated"
                        );
                    }
                    
                    opportunities.push(opportunity);
                }
                Ok(None) => {
                    trace!(target: "opportunity_evaluation::strategy", strategy = ?strategy_name, "No opportunity found for strategy");
                }
                Err(e) => {
                    // Keep error logs at warn level for visibility
                    warn!(
                        target: "opportunity_evaluation::error",
                        error = ?e,
                        strategy = ?strategy_name,
                        "Error evaluating with strategy"
                    );
                }
            }
        }
        
        // Always log the summary count at info level if opportunities found
        if !opportunities.is_empty() {
            info!(
                target: "opportunity_evaluation::summary",
                count = opportunities.len(),
                strategies = ?opportunities.iter().map(|o| &o.strategy).collect::<Vec<_>>(),
                "Found potential opportunities"
            );
        } else if should_log_detailed {
            debug!(target: "opportunity_evaluation::summary", "No opportunities found");
        }
        
        Ok(opportunities)
    }

    /// Creates the default set of scoring factors
    fn create_default_scoring_factors() -> Vec<Box<dyn ScoringFactor>> {
        vec![
            Box::new(ProfitFactor::new(0.3)),
            Box::new(TimeSensitivityFactor::new(0.2)),
            Box::new(RiskAdjustedReturnFactor::new(0.3)),
            Box::new(StrategySpecificFactor::new(0.2)),
        ]
    }

    /// Calculates the final opportunity score
    pub async fn calculate_opportunity_score(&self, opportunity: &MevOpportunity) -> f64 {
        // Get all scoring factors
        let scoring_factors = Self::create_default_scoring_factors();
        let mut total_weight = 0.0;
        let mut weighted_score = 0.0;
        
        // Only log detailed scoring calculation if detailed logging enabled
        let should_log_detailed = OPPORTUNITY_RATE_LOGGER.should_log() || 
            std::env::var("DETAILED_LOGGING").map(|v| v == "true").unwrap_or(false);
            
        for factor in &scoring_factors {
            let factor_score = factor.calculate_score(opportunity, &self.execution_thresholds).await;
            let factor_weight = factor.weight();
            
            if factor.is_critical() && factor_score <= 0.0 {
                if should_log_detailed {
                    debug!(
                        target: "opportunity_evaluation::scoring",
                        factor = factor.name(),
                        is_critical = true,
                        score = 0.0,
                        "Critical factor rejected opportunity"
                    );
                }
                return 0.0; // Critical factor failed
            }
            
            total_weight += factor_weight;
            weighted_score += factor_score * factor_weight;
            
            if should_log_detailed {
                trace!(
                    target: "opportunity_evaluation::scoring",
                    factor = factor.name(),
                    score = factor_score,
                    weight = factor_weight,
                    "Factor score calculated"
                );
            }
        }
        
        if total_weight <= 0.0 {
            return 0.0;
        }
        
        let final_score = weighted_score / total_weight;
        if should_log_detailed {
            debug!(
                target: "opportunity_evaluation::scoring",
                final_score,
                strategy = ?opportunity.strategy,
                "Calculated final opportunity score"
            );
        }
        
        final_score
    }

    /// Creates the default set of circuit breakers
    fn create_default_circuit_breakers() -> Vec<Box<dyn CircuitBreaker>> {
        vec![
            Box::new(VolatilityBreaker::default()),
            Box::new(GasPriceBreaker::default()),
            Box::new(MempoolBreaker::default()),
            Box::new(StrategyBreaker::default()),
        ]
    }

    /// Check all circuit breakers and return the most severe status
    pub async fn check_circuit_breakers(&self, conditions: &MarketConditions) -> CircuitBreakerStatus {
        let breakers = Self::create_default_circuit_breakers();
        let mut status = CircuitBreakerStatus::Normal;
        
        for breaker in breakers.iter() {
            let breaker_status = breaker.check(conditions).await;
            debug!(
                breaker = breaker.name(),
                status = ?breaker_status,
                "Circuit breaker check"
            );
            
            // Update to most severe status
            status = match (status, breaker_status) {
                (_, CircuitBreakerStatus::Critical) => CircuitBreakerStatus::Critical,
                (CircuitBreakerStatus::Normal, CircuitBreakerStatus::Warning) => CircuitBreakerStatus::Warning,
                (current, _) => current,
            };
            
            // Early exit on critical status from critical breakers
            if breaker.is_critical() && breaker_status == CircuitBreakerStatus::Critical {
                error!(
                    breaker = breaker.name(),
                    "Critical circuit breaker triggered - halting operations"
                );
                return CircuitBreakerStatus::Critical;
            }
        }
        
        status
    }

    /// Makes a go/no-go decision based on the opportunity, thresholds, and circuit breakers
    pub async fn make_execution_decision(&self, opportunity: &MevOpportunity) -> ExecutionDecision {
        // First check circuit breakers if market conditions are available
        if let Some(conditions) = self.get_market_conditions().await {
            let breaker_status = self.check_circuit_breakers(&conditions).await;
            match breaker_status {
                CircuitBreakerStatus::Critical => {
                    warn!("Circuit breaker status Critical - declining opportunity");
                    return ExecutionDecision::Decline;
                }
                CircuitBreakerStatus::Warning => {
                    // Increase scrutiny for warning status
                    if opportunity.confidence < self.min_confidence * 1.2 {
                        warn!("Increased confidence threshold not met under circuit breaker warning");
                        return ExecutionDecision::Decline;
                    }
                }
                _ => {}
            }
        }
        
        // Continue with normal decision logic...
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
        let opportunity_score = match opportunity.score {
            Some(score) => score,
            None => self.calculate_opportunity_score(opportunity).await
        };
            
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
    pub async fn should_execute(&self, opportunity: &MevOpportunity) -> bool {
        let decision = match opportunity.decision {
            Some(d) => d,
            None => self.make_execution_decision(opportunity).await
        };
            
        decision == ExecutionDecision::Execute
    }

    /// Starts monitoring an opportunity
    pub async fn monitor_opportunity(&self, opportunity: MevOpportunity) -> Result<()> {
        // Add to active opportunities
        self.active_opportunities.write().await.push(opportunity);
        
        Ok(())
    }

    // Fix the ProfitFactor implementation to handle temporary values correctly
    fn get_strategy_params(strategy: &MevStrategy) -> StrategyExecutionParams {
        match strategy {
            MevStrategy::Arbitrage => StrategyExecutionParams::default_for_arbitrage(),
            MevStrategy::Sandwich => StrategyExecutionParams::default_for_sandwich(),
            MevStrategy::TokenSnipe => StrategyExecutionParams::default_for_token_snipe(),
        }
    }

    /// Get current market conditions (to be implemented based on your data sources)
    async fn get_market_conditions(&self) -> Option<MarketConditions> {
        // TODO: Refactor market condition fetching.
        // This method relied on the removed `engine` field.
        // Market data should likely be fetched elsewhere (e.g., ListenBot or dedicated service)
        // and passed to the evaluator when needed (e.g., during scoring or decision making).
        warn!("get_market_conditions is currently disabled due to refactoring. Returning None.");
        None
        /*
        // Get market data from the engine
        if let Ok(prices) = self.engine.get_prices().await {
            let mut volatility_index: f64 = 0.0;
            let mut price_change_percent: f64 = 0.0;
            let mut volume_usd: f64 = 0.0;
            let mut liquidity_depth: f64 = 0.0;
            let mut mempool_density: u32 = 0;
            let mut gas_price_gwei: u64 = 0;

            // Calculate volatility and price changes from available price data
            if let Some(price_history) = self.engine.get_price_history().await.ok() {
                for (_, history) in price_history {
                    if history.len() >= 2 {
                        let latest = history.last().unwrap();
                        let previous = history.get(history.len() - 2).unwrap();
                        let change = ((latest - previous) / previous).abs();
                        volatility_index = if change * 100.0 > volatility_index {
                            change * 100.0
                        } else {
                            volatility_index
                        };
                        price_change_percent = if change * 100.0 > price_change_percent {
                            change * 100.0
                        } else {
                            price_change_percent
                        };
                    }
                }
            }

            // Get volume and liquidity data from prices
            for price in prices.values() {
                volume_usd += price * 1000.0; // Rough estimate based on price
                liquidity_depth += price * 10000.0; // Rough estimate based on price
            }

            // Get mempool stats from engine status
            if let Ok(status) = self.engine.get_status().await {
                mempool_density = status.pending_transactions as u32;
                gas_price_gwei = status.current_gas_price;
            }

            Some(MarketConditions {
                volatility_index,
                price_change_percent,
                volume_usd,
                liquidity_depth,
                mempool_density,
                gas_price_gwei,
                timestamp: chrono::Utc::now().timestamp(),
            })
        } else {
            None
        }
        */
    }

    /// Add RIG Agent for AI-powered decision enhancement
    pub fn with_rig_agent(mut self, rig_agent: Arc<RigAgent>) -> Self {
        self.rig_agent = Some(rig_agent);
        self
    }
    
    /// Sets the RIG agent for AI-enhanced decision making
    pub async fn set_rig_agent(&mut self, agent: Arc<RigAgent>) {
        self.rig_agent = Some(agent);
    }
    
    /// Enhances opportunity evaluation with AI insights from RIG Agent
    pub async fn enhance_with_ai(&self, opportunity: &mut MevOpportunity) -> Result<()> {
        // Skip if no RIG agent is set
        let Some(rig_agent) = &self.rig_agent else {
            debug!("No RIG Agent available for opportunity enhancement");
            return Ok(());
        };
        
        debug!(strategy = ?opportunity.strategy, "Enhancing opportunity evaluation with RIG Agent");
        
        // Create raw opportunity data from MevOpportunity
        let raw_opportunity = RawOpportunityData {
            source_dex: "Unknown".to_string(), // Could be improved by adding DEX info to MevOpportunity
            transaction_hash: opportunity.metadata.get("signature")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown_tx_hash")
                .to_string(),
            input_token: opportunity.involved_tokens.first()
                .cloned()
                .unwrap_or_else(|| "UNKNOWN".to_string()),
            output_token: opportunity.involved_tokens.get(1)
                .cloned()
                .unwrap_or_else(|| "UNKNOWN".to_string()),
            input_amount: opportunity.required_capital,
            output_amount: opportunity.required_capital + opportunity.estimated_profit,
        };
        
        // Get market context data - could be further enhanced by using real market data
        let market_context = MarketContext {
            input_token_price_usd: 1.0, // Placeholder - should use real price data
            output_token_price_usd: 1.0, // Placeholder
            pool_liquidity_usd: 1_000_000.0, // Placeholder
            recent_volatility_percent: 5.0, // Placeholder
        };
        
        // Get AI evaluation using RIG Agent
        let ai_evaluation = match rig_agent.evaluate_opportunity(&raw_opportunity, &market_context).await {
            Ok(eval) => eval,
            Err(e) => {
                warn!(error = ?e, "Failed to get AI evaluation from RIG Agent, using rule-based fallback");
                // Fallback to rule-based evaluation if AI fails
                rig_agent.evaluate_opportunity_rule_based(&raw_opportunity, &market_context)?
            }
        };
        
        // Enhance opportunity with AI insights
        let ai_confidence = ai_evaluation.confidence_score;
        
        // Average existing confidence with AI confidence, giving AI higher weight
        let combined_confidence = (opportunity.confidence * 0.3) + (ai_confidence * 0.7);
        opportunity.confidence = combined_confidence;
        
        // Update estimated profit if AI has a different view
        if (ai_evaluation.estimated_profit_usd - opportunity.estimated_profit).abs() > 1.0 {
            // If significant difference, choose more conservative estimate
            opportunity.estimated_profit = opportunity.estimated_profit.min(ai_evaluation.estimated_profit_usd);
            debug!(
                ai_profit = ai_evaluation.estimated_profit_usd,
                original_profit = opportunity.estimated_profit,
                final_profit = opportunity.estimated_profit,
                "Adjusted profit estimate based on AI insights"
            );
        }
        
        // Add AI reasoning to metadata
        let mut metadata = opportunity.metadata.clone();
        
        // Clone the string values before moving them
        let ai_reasoning = ai_evaluation.reasoning.clone();
        let ai_suggestion = ai_evaluation.suggested_action.clone();
        
        metadata.as_object_mut().map(|m| {
            m.insert("ai_reasoning".into(), serde_json::Value::String(ai_reasoning));
            m.insert("ai_suggestion".into(), serde_json::Value::String(ai_suggestion));
            let ai_conf_value = match serde_json::Number::from_f64(ai_confidence) {
                Some(num) => serde_json::Value::Number(num),
                None => serde_json::Value::Null,
            };
            m.insert("ai_confidence".into(), ai_conf_value);
        });
        opportunity.metadata = metadata;
        
        debug!(
            ai_confidence = ai_confidence,
            combined_confidence = combined_confidence,
            ai_suggested_action = %ai_evaluation.suggested_action,
            "Enhanced opportunity with AI insights"
        );
        
        Ok(())
    }
}

/// Factor that considers strategy-specific metrics
pub struct StrategySpecificFactor {
    weight: f64,
}

impl StrategySpecificFactor {
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }

    fn calculate_arbitrage_score(&self, opportunity: &MevOpportunity) -> f64 {
        let metadata: arbitrage::ArbitrageMetadata = 
            serde_json::from_value(opportunity.metadata.clone()).unwrap_or_else(|_| {
                warn!("Failed to parse arbitrage metadata");
                arbitrage::ArbitrageMetadata {
                    source_dex: "Unknown".to_string(),
                    target_dex: "Unknown".to_string(),
                    token_path: vec![],
                    price_difference_percent: 0.0,
                    estimated_gas_cost_usd: 0.0,
                    optimal_trade_size_usd: 0.0,
                    price_impact_percent: 0.0,
                }
            });
        
        // Calculate various subscores based on the arbitrage-specific metrics
        let price_diff_score = metadata.price_difference_percent / 3.0; // Normalize to 0-1 range (3% diff = 1.0)
        
        // Lower price impact is better
        let impact_score = (1.0 - metadata.price_impact_percent / 100.0 * 10.0).max(0.0); 
        
        // Simple calculation for now - score based on price difference and impact
        (price_diff_score * 0.7 + impact_score * 0.3).min(1.0)
    }

    fn calculate_sandwich_score(&self, opportunity: &MevOpportunity) -> f64 {
        let metadata: sandwich::SandwichMetadata = 
            serde_json::from_value(opportunity.metadata.clone()).unwrap_or_else(|_| {
                warn!("Failed to parse sandwich metadata");
                sandwich::SandwichMetadata {
                    dex: "Unknown".to_string(),
                    target_tx_hash: "Unknown".to_string(),
                    target_tx_value_usd: 0.0,
                    token_pair: ("Unknown".to_string(), "Unknown".to_string()),
                    pool_liquidity_usd: 0.0,
                    price_impact_pct: 0.0,
                    optimal_position_size_usd: 0.0,
                    frontrun_slippage_pct: 0.0,
                    backrun_slippage_pct: 0.0,
                }
            });
        
        // Calculate score based on transaction size (normalized to 0-1)
        let size_score = (metadata.target_tx_value_usd / 50000.0).min(1.0);
        
        // Total impact based on front and back run slippage
        let total_impact = metadata.frontrun_slippage_pct + metadata.backrun_slippage_pct;
        let impact_score = (1.0 - total_impact / 100.0 * 20.0).max(0.0);
        
        // Calculate gas-to-profit ratio (estimated via slippage)
        let gas_cost_est = opportunity.estimated_profit * 0.05; // assume 5% gas cost
        let gas_ratio = gas_cost_est / opportunity.estimated_profit;
        let gas_score = (1.0 - gas_ratio * 2.0).max(0.0);
        
        // Combine scores with weights
        (size_score * 0.3 + impact_score * 0.5 + gas_score * 0.2).min(1.0)
    }

    fn calculate_token_snipe_score(&self, opportunity: &MevOpportunity) -> f64 {
        let metadata: token_snipe::TokenSnipeMetadata = 
            serde_json::from_value(opportunity.metadata.clone()).unwrap_or_else(|_| {
                warn!("Failed to parse token snipe metadata");
                token_snipe::TokenSnipeMetadata {
                    token_address: "Unknown".to_string(),
                    token_symbol: "Unknown".to_string(),
                    dex: "Unknown".to_string(),
                    initial_liquidity_usd: 0.0,
                    initial_price_usd: 0.0,
                    initial_market_cap_usd: 0.0,
                    proposed_investment_usd: 0.0,
                    expected_return_multiplier: 0.0,
                    max_position_percent: 0.0,
                    optimal_hold_time_seconds: 0,
                    acquisition_supply_percent: 0.0,
                }
            });
        
        // Calculate various subscores for token sniping
        
        // Liquidity score (normalized to 0-1)
        let liquidity_score = (metadata.initial_liquidity_usd / 500000.0).min(1.0);
        
        // Market cap score (normalized to 0-1)
        let market_cap_score = (metadata.initial_market_cap_usd / 2000000.0).min(1.0);
        
        // Return multiplier score
        let return_score = (metadata.expected_return_multiplier / 10.0).min(1.0);
        
        // Position size relative to liquidity (lower is better)
        let position_score = (1.0 - metadata.max_position_percent / 10.0).max(0.0);
        
        // Combine scores with weights
        (liquidity_score * 0.3 + market_cap_score * 0.2 + return_score * 0.3 + position_score * 0.2).min(1.0)
    }
}

#[async_trait::async_trait]
impl ScoringFactor for StrategySpecificFactor {
    fn name(&self) -> &'static str {
        "strategy_specific"
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    async fn calculate_score(&self, opportunity: &MevOpportunity, _thresholds: &ExecutionThresholds) -> f64 {
        match opportunity.strategy {
            MevStrategy::Arbitrage => self.calculate_arbitrage_score(opportunity),
            MevStrategy::Sandwich => self.calculate_sandwich_score(opportunity),
            MevStrategy::TokenSnipe => self.calculate_token_snipe_score(opportunity),
        }
    }

    fn is_critical(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio;
    use anyhow::Result;

    // Mock strategy evaluator for testing
    struct MockEvaluator {
        strategy: MevStrategy,
        should_succeed: bool,
    }

    impl MockEvaluator {
        fn new(strategy: MevStrategy, should_succeed: bool) -> Self {
            Self {
                strategy,
                should_succeed,
            }
        }
    }

    #[async_trait::async_trait]
    impl MevStrategyEvaluator for MockEvaluator {
        fn strategy_type(&self) -> MevStrategy {
            self.strategy.clone()
        }

        async fn evaluate(&self, _data: &serde_json::Value) -> Result<Option<MevOpportunity>> {
            if self.should_succeed {
                Ok(Some(MevOpportunity {
                    strategy: self.strategy.clone(),
                    estimated_profit: 100.0,
                    confidence: 0.9,
                    risk_level: RiskLevel::Low,
                    required_capital: 1000.0,
                    execution_time: 500,
                    metadata: serde_json::json!({}),
                    score: None,
                    decision: None,
                    involved_tokens: Vec::new(),
                    allowed_output_tokens: Vec::new(),
                    allowed_programs: Vec::new(),
                    max_instructions: 0,
                }))
            } else {
                Ok(None)
            }
        }

        async fn validate(&self, _opportunity: &MevOpportunity) -> Result<bool> {
            Ok(self.should_succeed)
        }
    }

    #[test]
    fn test_strategy_registry_operations() {
        let mut registry = StrategyRegistry::new();
        
        // Test registration
        let evaluator = Box::new(MockEvaluator::new(MevStrategy::Arbitrage, true));
        assert!(registry.register(evaluator).is_none());
        
        // Test contains_strategy
        assert!(registry.contains_strategy(&MevStrategy::Arbitrage));
        assert!(!registry.contains_strategy(&MevStrategy::Sandwich));
        
        // Test get
        assert!(registry.get(&MevStrategy::Arbitrage).is_some());
        assert!(registry.get(&MevStrategy::Sandwich).is_none());
        
        // Test strategies list
        let strategies = registry.strategies();
        assert_eq!(strategies.len(), 1);
        assert_eq!(strategies[0], MevStrategy::Arbitrage);
        
        // Test remove
        let removed = registry.remove(&MevStrategy::Arbitrage);
        assert!(removed.is_some());
        assert!(!registry.contains_strategy(&MevStrategy::Arbitrage));
    }

    #[tokio::test]
    async fn test_evaluator_with_multiple_strategies() {
        let mut evaluator = create_test_evaluator().await;
        
        // Register multiple strategies
        evaluator.register_evaluator(Box::new(MockEvaluator::new(MevStrategy::Arbitrage, true)));
        evaluator.register_evaluator(Box::new(MockEvaluator::new(MevStrategy::Sandwich, true)));
        evaluator.register_evaluator(Box::new(MockEvaluator::new(MevStrategy::TokenSnipe, false)));
        
        // Test evaluation
        let test_data = serde_json::json!({
            "test": "data"
        });
        
        let opportunities = evaluator.evaluate_opportunity(test_data).await.unwrap();
        
        // Should get opportunities from Arbitrage and Sandwich, but not TokenSnipe
        assert_eq!(opportunities.len(), 2);
        
        // Verify opportunities have scores and decisions
        for opportunity in opportunities {
            assert!(opportunity.score.is_some());
            assert!(opportunity.decision.is_some());
            assert!(matches!(
                opportunity.strategy,
                MevStrategy::Arbitrage | MevStrategy::Sandwich
            ));
        }
    }

    #[tokio::test]
    async fn test_evaluator_threshold_filtering() {
        let mut evaluator = create_test_evaluator().await;
        
        // Create custom thresholds
        let mut thresholds = ExecutionThresholds::default();
        thresholds.min_confidence = 0.95; // Set high confidence requirement
        evaluator.update_execution_thresholds(thresholds);
        
        // Register a strategy that produces opportunities with 0.9 confidence
        evaluator.register_evaluator(Box::new(MockEvaluator::new(MevStrategy::Arbitrage, true)));
        
        let test_data = serde_json::json!({
            "test": "data"
        });
        
        let opportunities = evaluator.evaluate_opportunity(test_data).await.unwrap();
        
        // Should get no opportunities due to confidence threshold
        assert_eq!(opportunities.len(), 0);
    }

    #[tokio::test]
    async fn test_strategy_replacement() {
        let mut evaluator = create_test_evaluator().await;
        
        // Register initial strategy
        evaluator.register_evaluator(Box::new(MockEvaluator::new(MevStrategy::Arbitrage, false)));
        
        // Test with initial strategy
        let test_data = serde_json::json!({
            "test": "data"
        });
        let opportunities = evaluator.evaluate_opportunity(test_data.clone()).await.unwrap();
        assert_eq!(opportunities.len(), 0);
        
        // Replace with new strategy that succeeds
        evaluator.register_evaluator(Box::new(MockEvaluator::new(MevStrategy::Arbitrage, true)));
        
        // Test with replaced strategy
        let opportunities = evaluator.evaluate_opportunity(test_data).await.unwrap();
        assert_eq!(opportunities.len(), 1);
    }

    // Helper to create evaluator for tests (panics on error for simplicity)
    async fn create_test_evaluator() -> OpportunityEvaluator {
        let (engine_instance, _) = Engine::from_env().await
            .expect("Engine creation failed in test - check environment variables");
        OpportunityEvaluator {
            // engine: Arc::new(engine_instance), // Removed engine field assignment
            strategy_registry: StrategyRegistry::new(),
            min_confidence: 0.7, // Default test confidence
            max_risk_level: RiskLevel::High, // Default test risk level
            min_profit_threshold: 50.0, // Default test profit threshold
            active_opportunities: Arc::new(RwLock::new(Vec::new())),
            execution_thresholds: ExecutionThresholds::default(),
            strategy_executor: RwLock::new(None), // Initialize executor as None
            rig_agent: None,
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
            involved_tokens: vec!["SOL".to_string(), "USDC".to_string()],
            allowed_output_tokens: vec!["SOL".to_string()],
            allowed_programs: vec!["DEX1".to_string()],
            max_instructions: 4,
        };

        assert!(opportunity.confidence >= evaluator.min_confidence);
        assert!((opportunity.risk_level.clone() as u8) <= (evaluator.max_risk_level.clone() as u8));
        assert!(opportunity.estimated_profit >= evaluator.min_profit_threshold);
    }
    
    #[tokio::test]
    async fn test_opportunity_scoring() {
        let evaluator = create_test_evaluator().await;
        let opportunity = create_test_opportunity(
            MevStrategy::Arbitrage,
            100.0,  // Good profit
            0.9,    // High confidence
            RiskLevel::Low,
            50.0,   // Low capital requirement
            1000,   // Fast execution
        );
        
        let score = evaluator.calculate_opportunity_score(&opportunity).await;
        assert!(score > 0.8, "High quality opportunity should score > 0.8");
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
            involved_tokens: vec!["SOL".to_string(), "USDC".to_string()],
            allowed_output_tokens: vec!["SOL".to_string()],
            allowed_programs: vec!["DEX1".to_string()],
            max_instructions: 4,
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
            involved_tokens: vec!["SOL".to_string(), "USDC".to_string()],
            allowed_output_tokens: vec!["SOL".to_string()],
            allowed_programs: vec!["DEX1".to_string()],
            max_instructions: 4,
        };
        
        // Should hold - medium score
        let medium_opportunity = MevOpportunity {
            strategy: MevStrategy::Arbitrage,
            estimated_profit: 100.0,
            confidence: 0.85,
            risk_level: RiskLevel::Medium,
            required_capital: 1500.0,
            execution_time: 400,
            metadata: serde_json::json!({}),
            score: Some(0.6), // Medium score (between 0.5 and 0.7)
            decision: None,
            involved_tokens: vec!["SOL".to_string(), "USDC".to_string()],
            allowed_output_tokens: vec!["SOL".to_string()],
            allowed_programs: vec!["DEX1".to_string()],
            max_instructions: 4,
        };

        let decision = evaluator.make_execution_decision(&good_opportunity).await;
        assert_eq!(decision, ExecutionDecision::Execute);

        let decision = evaluator.make_execution_decision(&risky_opportunity).await;
        assert_eq!(decision, ExecutionDecision::Decline);

        let decision = evaluator.make_execution_decision(&medium_opportunity).await;
        assert_eq!(decision, ExecutionDecision::Hold);
    }

    #[tokio::test]
    async fn test_profit_factor() {
        let factor = ProfitFactor { weight: 0.4 };
        let thresholds = ExecutionThresholds::default();
        
        // Test opportunity with profit exactly at minimum
        let opportunity = create_test_opportunity(
            MevStrategy::Arbitrage,
            20.0,  // min_profit for arbitrage
            0.9,
            RiskLevel::Low,
            1000.0,
            500,
        );
        let score = factor.calculate_score(&opportunity, &thresholds).await;
        assert_eq!(score, 0.0);
        
        // Test opportunity with profit 2x minimum
        let opportunity = create_test_opportunity(
            MevStrategy::Arbitrage,
            40.0,
            0.9,
            RiskLevel::Low,
            1000.0,
            500,
        );
        let score = factor.calculate_score(&opportunity, &thresholds).await;
        assert!(score > 0.0 && score < 1.0);
        
        // Test opportunity with profit 4x minimum
        let opportunity = create_test_opportunity(
            MevStrategy::Arbitrage,
            80.0,
            0.9,
            RiskLevel::Low,
            1000.0,
            500,
        );
        let score = factor.calculate_score(&opportunity, &thresholds).await;
        assert_eq!(score, 1.0);
    }
    
    #[tokio::test]
    async fn test_time_sensitivity_factor() {
        let factor = TimeSensitivityFactor { weight: 0.2 };
        let thresholds = ExecutionThresholds::default();
        
        // Test opportunity with execution time at maximum
        let opportunity = create_test_opportunity(
            MevStrategy::Arbitrage,
            100.0,
            0.9,
            RiskLevel::Low,
            1000.0,
            2000,  // max_execution_time
        );
        let score = factor.calculate_score(&opportunity, &thresholds).await;
        assert_eq!(score, 0.0);
        
        // Test opportunity with execution time at half maximum
        let opportunity = create_test_opportunity(
            MevStrategy::Arbitrage,
            100.0,
            0.9,
            RiskLevel::Low,
            1000.0,
            1000,
        );
        let score = factor.calculate_score(&opportunity, &thresholds).await;
        assert_eq!(score, 0.5);
        
        // Test opportunity with very fast execution time
        let opportunity = create_test_opportunity(
            MevStrategy::Arbitrage,
            100.0,
            0.9,
            RiskLevel::Low,
            1000.0,
            200,
        );
        let score = factor.calculate_score(&opportunity, &thresholds).await;
        assert_eq!(score, 0.9);
    }
    
    #[tokio::test]
    async fn test_risk_adjusted_return_factor() {
        let factor = RiskAdjustedReturnFactor { weight: 0.4 };
        let thresholds = ExecutionThresholds::default();
        
        // Test low risk opportunity with good profit ratio
        let opportunity = create_test_opportunity(
            MevStrategy::Arbitrage,
            150.0,  // 15% return
            0.9,
            RiskLevel::Low,
            1000.0,
            500,
        );
        let score = factor.calculate_score(&opportunity, &thresholds).await;
        assert!(score > 0.8);
        
        // Test high risk opportunity with same profit ratio
        let opportunity = create_test_opportunity(
            MevStrategy::Arbitrage,
            150.0,  // 15% return
            0.9,
            RiskLevel::High,
            1000.0,
            500,
        );
        let score = factor.calculate_score(&opportunity, &thresholds).await;
        assert!(score < 0.5);
        
        // Test opportunity below minimum profit-risk ratio
        let opportunity = create_test_opportunity(
            MevStrategy::Arbitrage,
            10.0,  // 1% return
            0.9,
            RiskLevel::Medium,
            1000.0,
            500,
        );
        let score = factor.calculate_score(&opportunity, &thresholds).await;
        assert_eq!(score, 0.0);
    }
    
    fn create_test_opportunity(
        strategy: MevStrategy,
        estimated_profit: f64,
        confidence: f64,
        risk_level: RiskLevel,
        required_capital: f64,
        execution_time: u64,
    ) -> MevOpportunity {
        MevOpportunity {
            strategy,
            estimated_profit,
            confidence,
            risk_level,
            required_capital,
            execution_time,
            metadata: serde_json::json!({}),
            score: None,
            decision: None,
            involved_tokens: Vec::new(),
            allowed_output_tokens: Vec::new(),
            allowed_programs: Vec::new(),
            max_instructions: 0,
        }
    }

    fn create_test_market_conditions(
        volatility: f64,
        gas_price: u64,
        mempool_density: u32,
    ) -> MarketConditions {
        MarketConditions {
            volatility_index: volatility,
            price_change_percent: 0.0,  // Not used in current tests
            volume_usd: 1_000_000.0,    // Default test value
            liquidity_depth: 500_000.0,  // Default test value
            mempool_density,
            gas_price_gwei: gas_price,
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    #[tokio::test]
    async fn test_volatility_breaker() {
        let breaker = VolatilityBreaker::default();
        
        // Test normal conditions
        let conditions = create_test_market_conditions(20.0, 100, 20);
        assert_eq!(breaker.check(&conditions).await, CircuitBreakerStatus::Normal);
        
        // Test warning conditions
        let conditions = create_test_market_conditions(30.0, 100, 20);
        assert_eq!(breaker.check(&conditions).await, CircuitBreakerStatus::Warning);
        
        // Test critical conditions
        let conditions = create_test_market_conditions(45.0, 100, 20);
        assert_eq!(breaker.check(&conditions).await, CircuitBreakerStatus::Critical);
    }

    #[tokio::test]
    async fn test_gas_price_breaker() {
        let breaker = GasPriceBreaker::default();
        
        // Test normal conditions
        let conditions = create_test_market_conditions(10.0, 150, 20);
        assert_eq!(breaker.check(&conditions).await, CircuitBreakerStatus::Normal);
        
        // Test warning conditions
        let conditions = create_test_market_conditions(10.0, 300, 20);
        assert_eq!(breaker.check(&conditions).await, CircuitBreakerStatus::Warning);
        
        // Test critical conditions
        let conditions = create_test_market_conditions(10.0, 600, 20);
        assert_eq!(breaker.check(&conditions).await, CircuitBreakerStatus::Critical);
    }

    #[tokio::test]
    async fn test_mempool_breaker() {
        let breaker = MempoolBreaker::default();
        
        // Test normal conditions
        let conditions = create_test_market_conditions(10.0, 100, 30);
        assert_eq!(breaker.check(&conditions).await, CircuitBreakerStatus::Normal);
        
        // Test warning conditions
        let conditions = create_test_market_conditions(10.0, 100, 75);
        assert_eq!(breaker.check(&conditions).await, CircuitBreakerStatus::Warning);
        
        // Test critical conditions
        let conditions = create_test_market_conditions(10.0, 100, 150);
        assert_eq!(breaker.check(&conditions).await, CircuitBreakerStatus::Critical);
    }

    #[tokio::test]
    async fn test_circuit_breaker_integration() {
        let evaluator = create_test_evaluator().await;
        
        // Test all normal conditions
        let conditions = create_test_market_conditions(20.0, 150, 30);
        assert_eq!(evaluator.check_circuit_breakers(&conditions).await, CircuitBreakerStatus::Normal);
        
        // Test warning from one breaker
        let conditions = create_test_market_conditions(30.0, 150, 30);
        assert_eq!(evaluator.check_circuit_breakers(&conditions).await, CircuitBreakerStatus::Warning);
        
        // Test critical from non-critical breaker
        let conditions = create_test_market_conditions(20.0, 600, 30);
        assert_eq!(evaluator.check_circuit_breakers(&conditions).await, CircuitBreakerStatus::Critical);
        
        // Test critical from critical breaker (volatility)
        let conditions = create_test_market_conditions(45.0, 150, 30);
        assert_eq!(evaluator.check_circuit_breakers(&conditions).await, CircuitBreakerStatus::Critical);
    }

    #[tokio::test]
    async fn test_decision_making_with_circuit_breakers() {
        let evaluator = create_test_evaluator().await;
        let opportunity = create_test_opportunity(
            MevStrategy::Arbitrage,
            100.0,
            0.9,
            RiskLevel::Low,
            1000.0,
            500,
        );
        
        // Test decision with critical market conditions
        let conditions = create_test_market_conditions(45.0, 150, 30);
        assert_eq!(
            evaluator.make_execution_decision(&opportunity).await,
            ExecutionDecision::Decline
        );
        
        // Test decision with warning conditions and high confidence
        let conditions = create_test_market_conditions(30.0, 150, 30);
        let mut high_confidence_opportunity = opportunity.clone();
        high_confidence_opportunity.confidence = 0.95;
        assert_ne!(
            evaluator.make_execution_decision(&high_confidence_opportunity).await,
            ExecutionDecision::Decline
        );
        
        // Test decision with warning conditions and lower confidence
        let mut low_confidence_opportunity = opportunity.clone();
        low_confidence_opportunity.confidence = 0.75;
        assert_eq!(
            evaluator.make_execution_decision(&low_confidence_opportunity).await,
            ExecutionDecision::Decline
        );
    }
} 