use super::pipeline::{Condition, ConditionType};
use crate::engine::EngineError;
use crate::market_data::MarketData;
use std::collections::HashMap;
use anyhow::Result;

pub struct Evaluator;

#[derive(Debug, thiserror::Error)]
pub enum EvaluatorError {
    #[error("[Evaluator] Failed to evaluate conditions: {0}")]
    EvaluateConditionsError(String),

    #[error("[Evaluator] Failed to evaluate price condition: {0}")]
    PriceEvaluationError(String),

    #[error("[Evaluator] Missing price data for asset: {0}")]
    MissingPriceData(String),

    #[error("[Evaluator] Invalid condition type: {0}")]
    InvalidConditionType(String),

    #[error("[Evaluator] Market data error: {0}")]
    MarketDataError(String),
}

impl From<EvaluatorError> for EngineError {
    fn from(err: EvaluatorError) -> Self {
        EngineError::EvaluatePipelineError(err)
    }
}

#[derive(Debug)]
pub struct EvaluationContext {
    pub prices: HashMap<String, f64>,
    pub market_data: HashMap<String, MarketData>,
}

impl Evaluator {
    pub fn evaluate_conditions(
        conditions: &[Condition],
        context: &EvaluationContext,
    ) -> Result<bool, EvaluatorError> {
        conditions.iter().try_fold(true, |acc, c| {
            Ok(acc && Self::evaluate_condition(c, context)?)
        })
    }

    fn evaluate_condition(
        condition: &Condition,
        context: &EvaluationContext,
    ) -> Result<bool, EvaluatorError> {
        match condition.condition_type {
            ConditionType::PriceAbove { asset, threshold } => {
                let price = context.prices.get(&asset).ok_or_else(|| {
                    EvaluatorError::MissingPriceData(asset.clone())
                })?;
                Ok(*price > threshold)
            }
            ConditionType::PriceBelow { asset, threshold } => {
                let price = context.prices.get(&asset).ok_or_else(|| {
                    EvaluatorError::MissingPriceData(asset.clone())
                })?;
                Ok(*price < threshold)
            }
            ConditionType::VolumeAbove { asset, threshold } => {
                let market_data = context.market_data.get(&asset).ok_or_else(|| {
                    EvaluatorError::MarketDataError(format!("No market data for {}", asset))
                })?;
                Ok(market_data.volume_24h > threshold)
            }
            ConditionType::VolatilityBelow { asset, threshold } => {
                let market_data = context.market_data.get(&asset).ok_or_else(|| {
                    EvaluatorError::MarketDataError(format!("No market data for {}", asset))
                })?;
                Ok(market_data.volatility < threshold)
            }
            ConditionType::LiquidityAbove { asset, threshold } => {
                let market_data = context.market_data.get(&asset).ok_or_else(|| {
                    EvaluatorError::MarketDataError(format!("No market data for {}", asset))
                })?;
                Ok(market_data.liquidity > threshold)
            }
            ConditionType::MarketCapAbove { asset, threshold } => {
                let market_data = context.market_data.get(&asset).ok_or_else(|| {
                    EvaluatorError::MarketDataError(format!("No market data for {}", asset))
                })?;
                Ok(market_data.market_cap > threshold)
            }
            ConditionType::PriceChangeAbove { asset, threshold } => {
                let market_data = context.market_data.get(&asset).ok_or_else(|| {
                    EvaluatorError::MarketDataError(format!("No market data for {}", asset))
                })?;
                Ok(market_data.price_change_24h > threshold)
            }
            _ => Err(EvaluatorError::InvalidConditionType(format!(
                "Unsupported condition type: {:?}",
                condition.condition_type
            ))),
        }
    }
} 