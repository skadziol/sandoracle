use super::evaluator::{Evaluator, EvaluationContext};
use crate::engine::Engine;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    PriceAbove {
        asset: String,
        threshold: f64,
    },
    PriceBelow {
        asset: String,
        threshold: f64,
    },
    VolumeAbove {
        asset: String,
        threshold: f64,
    },
    VolatilityBelow {
        asset: String,
        threshold: f64,
    },
    LiquidityAbove {
        asset: String,
        threshold: f64,
    },
    MarketCapAbove {
        asset: String,
        threshold: f64,
    },
    PriceChangeAbove {
        asset: String,
        threshold: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub condition_type: ConditionType,
    pub description: String,
}

#[derive(Debug)]
pub struct Pipeline {
    pub id: String,
    pub conditions: Vec<Condition>,
    pub engine: Engine,
}

impl Pipeline {
    pub fn new(id: String, conditions: Vec<Condition>, engine: Engine) -> Self {
        Self {
            id,
            conditions,
            engine,
        }
    }

    pub async fn evaluate(&self) -> Result<bool> {
        // Get current prices
        let prices = self.engine.get_prices().await?;
        
        // Get market data for all assets
        let market_data = self.engine.get_all_market_data().await?;

        let context = EvaluationContext {
            prices,
            market_data,
        };

        Ok(Evaluator::evaluate_conditions(&self.conditions, &context)?)
    }
}

#[derive(Debug, Clone)]
pub struct Status {
    pub id: String,
    pub is_active: bool,
    pub last_evaluation: Option<bool>,
    pub last_update: u64,
} 