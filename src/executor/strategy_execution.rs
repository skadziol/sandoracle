use crate::evaluator::{MevOpportunity, MevStrategy, RiskLevel, StrategyExecutionService};
use crate::executor::{TransactionExecutor, strategies::StrategyExecutor};
use crate::error::{Result, SandoError};
use solana_sdk::signature::Signature;
use async_trait::async_trait;
use tracing::{info, error, warn, debug};
use solana_sdk::transaction::Transaction;
use std::sync::Arc;

/// Implementation of the StrategyExecutionService that delegates to the appropriate
/// strategy-specific executor based on the opportunity type
pub struct ExecutionService {
    /// The strategy executor that handles the actual transaction building and execution
    strategy_executor: StrategyExecutor,
    /// Maximum number of retries for failed transactions
    max_retries: u32,
    /// Whether to simulate transactions before execution
    simulate_first: bool,
}

impl ExecutionService {
    /// Create a new ExecutionService with the given TransactionExecutor
    pub fn new(transaction_executor: TransactionExecutor) -> Self {
        Self {
            strategy_executor: StrategyExecutor::new(transaction_executor),
            max_retries: 3,
            simulate_first: true,
        }
    }

    /// Set the maximum number of retries for transaction execution
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Enable or disable transaction simulation before execution
    pub fn with_simulation(mut self, simulate: bool) -> Self {
        self.simulate_first = simulate;
        self
    }

    /// Get a reference to the underlying TransactionExecutor
    pub fn transaction_executor(&self) -> &TransactionExecutor {
        self.strategy_executor.get_executor()
    }

    /// Validate the opportunity before execution
    #[cfg(test)]
    pub async fn validate_opportunity(&self, opportunity: &MevOpportunity) -> Result<bool> {
        // Skip validation for simulation mode
        if self.transaction_executor().simulation_mode() {
            return Ok(true);
        }

        // Check if opportunity has minimum required profit
        if opportunity.estimated_profit <= 0.0 {
            warn!(
                strategy = ?opportunity.strategy,
                estimated_profit = opportunity.estimated_profit,
                "Skipping opportunity execution: non-positive estimated profit"
            );
            return Ok(false);
        }

        // Check risk level
        if opportunity.risk_level == RiskLevel::High {
            warn!(
                strategy = ?opportunity.strategy,
                risk_level = ?opportunity.risk_level,
                "Executing high-risk opportunity: consider additional validation"
            );
        }

        // Check confidence
        if opportunity.confidence < 0.7 {
            warn!(
                strategy = ?opportunity.strategy,
                confidence = opportunity.confidence,
                "Low confidence opportunity: proceed with caution"
            );
        }

        // Additional strategy-specific validation
        match opportunity.strategy {
            MevStrategy::Arbitrage => {
                // Check that there are tokens involved
                if opportunity.involved_tokens.is_empty() {
                    warn!("Arbitrage opportunity has no tokens listed");
                    return Ok(false);
                }
            },
            MevStrategy::Sandwich => {
                // Check that there are tokens involved and the opportunity hasn't expired
                if opportunity.involved_tokens.is_empty() {
                    warn!("Sandwich opportunity has no tokens listed");
                    return Ok(false);
                }
                
                // Parse metadata to check target transaction
                if let Ok(metadata) = serde_json::from_value::<crate::executor::strategies::SandwichMetadata>(opportunity.metadata.clone()) {
                    if metadata.target_tx_hash.is_empty() {
                        warn!("Sandwich opportunity has no target transaction hash");
                        return Ok(false);
                    }
                } else {
                    warn!("Failed to parse sandwich metadata");
                    return Ok(false);
                }
            },
            MevStrategy::TokenSnipe => {
                // Check that there are tokens involved
                if opportunity.involved_tokens.is_empty() {
                    warn!("Token snipe opportunity has no tokens listed");
                    return Ok(false);
                }
            },
        }

        Ok(true)
    }

    #[cfg(not(test))]
    async fn validate_opportunity(&self, opportunity: &MevOpportunity) -> Result<bool> {
        // Skip validation for simulation mode
        if self.transaction_executor().simulation_mode() {
            return Ok(true);
        }

        // Check if opportunity has minimum required profit
        if opportunity.estimated_profit <= 0.0 {
            warn!(
                strategy = ?opportunity.strategy,
                estimated_profit = opportunity.estimated_profit,
                "Skipping opportunity execution: non-positive estimated profit"
            );
            return Ok(false);
        }

        // Check risk level
        if opportunity.risk_level == RiskLevel::High {
            warn!(
                strategy = ?opportunity.strategy,
                risk_level = ?opportunity.risk_level,
                "Executing high-risk opportunity: consider additional validation"
            );
        }

        // Check confidence
        if opportunity.confidence < 0.7 {
            warn!(
                strategy = ?opportunity.strategy,
                confidence = opportunity.confidence,
                "Low confidence opportunity: proceed with caution"
            );
        }

        // Additional strategy-specific validation
        match opportunity.strategy {
            MevStrategy::Arbitrage => {
                // Check that there are tokens involved
                if opportunity.involved_tokens.is_empty() {
                    warn!("Arbitrage opportunity has no tokens listed");
                    return Ok(false);
                }
            },
            MevStrategy::Sandwich => {
                // Check that there are tokens involved and the opportunity hasn't expired
                if opportunity.involved_tokens.is_empty() {
                    warn!("Sandwich opportunity has no tokens listed");
                    return Ok(false);
                }
                
                // Parse metadata to check target transaction
                if let Ok(metadata) = serde_json::from_value::<crate::executor::strategies::SandwichMetadata>(opportunity.metadata.clone()) {
                    if metadata.target_tx_hash.is_empty() {
                        warn!("Sandwich opportunity has no target transaction hash");
                        return Ok(false);
                    }
                } else {
                    warn!("Failed to parse sandwich metadata");
                    return Ok(false);
                }
            },
            MevStrategy::TokenSnipe => {
                // Check that there are tokens involved
                if opportunity.involved_tokens.is_empty() {
                    warn!("Token snipe opportunity has no tokens listed");
                    return Ok(false);
                }
            },
        }

        Ok(true)
    }

    /// Execute an opportunity with retry logic
    async fn execute_with_retry(&self, opportunity: &MevOpportunity) -> Result<Signature> {
        let mut attempts = 0;
        let max_attempts = self.max_retries + 1; // +1 for initial attempt
        
        while attempts < max_attempts {
            attempts += 1;
            
            match self.strategy_executor.execute_opportunity(opportunity).await {
                Ok(signature) => {
                    info!(
                        strategy = ?opportunity.strategy,
                        signature = %signature,
                        "Transaction execution successful"
                    );
                    return Ok(signature);
                },
                Err(err) => {
                    if attempts < max_attempts {
                        warn!(
                            strategy = ?opportunity.strategy,
                            error = %err,
                            attempt = attempts,
                            max_attempts = max_attempts,
                            "Transaction execution failed, retrying..."
                        );
                        
                        // Exponential backoff: 500ms, 1000ms, 2000ms, ...
                        let delay = std::time::Duration::from_millis((500u64) * (2u64.pow(attempts as u32 - 1)));
                        tokio::time::sleep(delay).await;
                    } else {
                        error!(
                            strategy = ?opportunity.strategy,
                            error = %err,
                            "Transaction execution failed after all retry attempts"
                        );
                        return Err(err);
                    }
                }
            }
        }
        
        // This should never be reached due to the return in the last error case above
        Err(SandoError::InternalError("Unexpected error in execute_with_retry".to_string()))
    }
}

#[async_trait]
impl StrategyExecutionService for ExecutionService {
    async fn execute_opportunity(&self, opportunity: &MevOpportunity) -> anyhow::Result<Signature> {
        info!(
            strategy = ?opportunity.strategy,
            estimated_profit = opportunity.estimated_profit,
            risk_level = ?opportunity.risk_level,
            "Starting opportunity execution"
        );
        
        // Validate the opportunity
        match self.validate_opportunity(opportunity).await {
            Ok(true) => {
                debug!("Opportunity validation passed, proceeding with execution");
            },
            Ok(false) => {
                warn!("Opportunity validation failed, aborting execution");
                return Err(anyhow::anyhow!("Opportunity validation failed"));
            },
            Err(err) => {
                error!(error = %err, "Error during opportunity validation");
                return Err(anyhow::anyhow!("Validation error: {}", err));
            }
        }
        
        // Execute with retry logic
        match self.execute_with_retry(opportunity).await {
            Ok(signature) => Ok(signature),
            Err(err) => Err(anyhow::anyhow!("Execution error: {}", err)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::{MevStrategy, RiskLevel};
    
    // Mock test for ExecutionService
    #[tokio::test]
    async fn test_validate_opportunity() {
        // This is a placeholder for actual tests
        // In a real test, you would create a mock TransactionExecutor
        // and test the validate_opportunity method
    }
} 