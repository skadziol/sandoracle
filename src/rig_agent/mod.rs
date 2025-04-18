use crate::error::{Result, SandoError, RigErrorKind};
use rig::providers::anthropic::{Client as AnthropicClient, completion::CompletionModel as AnthropicCompletionModel}; // Corrected import path for the concrete model type
use rig::completion::{Prompt, PromptError}; // Import Prompt trait and its error
use rig::agent::Agent; // Import Agent
use tracing::{info, error, warn}; // Added warn
use std::sync::Arc;
use serde::{Deserialize, Serialize}; // For parsing LLM response
use serde_json; // For JSON handling
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::HashMap;

/// Interface for the RIG AI Agent
#[derive(Clone)]
pub struct RigAgent {
    agent: Arc<Agent<AnthropicCompletionModel>>, // Store the Agent wrapping the concrete model
    model_name: String,
}

// Placeholder for data coming from Listen Bot / Opportunity source
#[derive(Debug, Clone, Serialize)] 
pub struct RawOpportunityData {
    pub source_dex: String,
    pub transaction_hash: String, // Example field
    pub input_token: String,
    pub output_token: String,
    pub input_amount: f64,
    pub output_amount: f64,
    // Add other relevant fields from the transaction
}

// Placeholder for Market Context data
#[derive(Debug, Clone, Serialize)]
pub struct MarketContext {
    pub input_token_price_usd: f64,
    pub output_token_price_usd: f64,
    pub pool_liquidity_usd: f64,
    pub recent_volatility_percent: f64, // Example field
    
    // Enhanced market data fields
    #[serde(default)]
    pub market_token_prices: HashMap<String, f64>,
    
    #[serde(default)]
    pub market_token_volatility: HashMap<String, f64>,
    
    #[serde(default)]
    pub market_token_price_changes: HashMap<String, f64>,
    
    #[serde(default)]
    pub market_token_forecasts: HashMap<String, f64>,
    
    #[serde(default = "default_f64")]
    pub average_market_price: f64,
    
    #[serde(default = "default_f64")]
    pub average_market_volatility: f64,
    
    #[serde(default = "default_f64")]
    pub average_market_price_change: f64,
}

fn default_f64() -> f64 {
    0.0
}

/// Represents the structured evaluation expected from the RIG agent
#[derive(Debug, Clone, Deserialize)]
pub struct OpportunityEvaluation {
    pub is_viable: bool,
    pub confidence_score: f64, // Scale 0.0 to 1.0
    pub estimated_profit_usd: f64,
    pub reasoning: String, // LLM's explanation
    pub suggested_action: String, // e.g., "Execute Arbitrage", "Monitor", "Decline"
    pub error: Option<String>, // If LLM indicates an error during its evaluation
}

impl RigAgent {
    /// Creates a new RigAgent instance from environment variables.
    /// Expects ANTHROPIC_API_KEY and optionally RIG_AGENT_MODEL (defaults to Claude Sonnet 3.5).
    pub fn from_env() -> Result<Self> {
        info!("Initializing RigAgent from environment...");

        let client = AnthropicClient::from_env(); 
        // TODO: Add error handling if from_env panics or returns Option

        let model_name = std::env::var("RIG_AGENT_MODEL")
            .unwrap_or_else(|_| "claude-3-5-sonnet-20240620".to_string());
        
        info!(model = %model_name, "Building RIG agent...");

        // Build the agent using the client and model name
        // build() might return an error or the Agent directly, assuming direct for now
        let agent = client.agent(&model_name).build(); 
        // TODO: Handle potential error from build()

        info!("RigAgent initialized successfully.");
        Ok(Self {
            agent: Arc::new(agent),
            model_name,
        })
    }

    /// Evaluates an opportunity using the RIG agent, incorporating market context.
    pub async fn evaluate_opportunity(
        &self,
        raw_opportunity: &RawOpportunityData,
        market_context: &MarketContext,
    ) -> Result<OpportunityEvaluation> {
        info!(model = %self.model_name, opportunity_hash = %raw_opportunity.transaction_hash, "Evaluating opportunity with RIG agent...");

        // --- 1. Construct Prompt --- 
        // Use raw string literal r#"..."# to handle quotes easily.
        // Escape internal JSON quotes with \"
        let prompt_text = format!(
            r#"Analyze the following potential Solana MEV opportunity based on the provided context.

Opportunity Data:
{}

Market Context:
{}

Enhanced Market Analysis:
- Token Price Forecasts (5 min): {}
- Average Market Volatility: {:.2}%
- Price Change 24h: {:.2}%
- Market Pattern: {}

Instructions:
1. Assess if this is a potentially viable MEV opportunity (e.g., arbitrage, sandwich potential).
2. Estimate the potential profit in USD, considering swap amounts, prices, and the price forecasts.
3. Provide a confidence score (0.0 to 1.0) for the viability and profitability estimate.
4. Briefly explain your reasoning, including consideration of the price forecasts and market volatility.
5. Suggest an action: Execute Arbitrage, Execute Sandwich, Monitor, Decline.
6. Respond ONLY with a valid JSON object containing these fields: "is_viable" (boolean), "confidence_score" (float), "estimated_profit_usd" (float), "reasoning" (string), "suggested_action" (string), "error" (string or null).
Example JSON: {{"is_viable\": true, \"confidence_score\": 0.85, \"estimated_profit_usd\": 150.50, \"reasoning\": \"Significant price difference observed...\", \"suggested_action\": \"Execute Arbitrage\", \"error\": null}}"#,
            serde_json::to_string_pretty(raw_opportunity).unwrap_or_else(|_| "Invalid opportunity data".to_string()),
            serde_json::to_string_pretty(market_context).unwrap_or_else(|_| "Invalid market context".to_string()),
            // Format forecasts as a readable string
            market_context.market_token_forecasts.iter()
                .map(|(token, &forecast)| format!("{}: ${:.4}", token, forecast))
                .collect::<Vec<_>>()
                .join(", "),
            market_context.recent_volatility_percent,
            market_context.average_market_price_change,
            // Provide a simple market pattern description based on volatility and price changes
            if market_context.average_market_volatility > 0.05 && market_context.average_market_price_change > 2.0 {
                "Volatile uptrend"
            } else if market_context.average_market_volatility > 0.05 && market_context.average_market_price_change < -2.0 {
                "Volatile downtrend"
            } else if market_context.average_market_volatility > 0.05 {
                "High volatility, sideways"
            } else if market_context.average_market_price_change > 2.0 {
                "Stable uptrend"
            } else if market_context.average_market_price_change < -2.0 {
                "Stable downtrend"
            } else {
                "Stable, sideways"
            }
        );

        info!(prompt_length = prompt_text.len(), "Constructed prompt for RIG agent");

        // --- 2. Call RIG Agent --- 
        let response_string = self.agent
            .prompt(prompt_text.as_str()) // Pass as &str
            .await
            .map_err(|e: PromptError| { 
                error!(error = ?e, "RIG agent prompt failed");
                match e {
                    PromptError::CompletionError(comp_err) => 
                        SandoError::rig(RigErrorKind::ModelError, format!("Agent completion failed: {}", comp_err)),
                    PromptError::ToolError(tool_err) => 
                        SandoError::rig(RigErrorKind::Other, format!("Agent tool error: {}", tool_err)),
                }
            })?;
        
        info!(response_length = response_string.len(), "Received response string from RIG agent.");

        // --- 3. Parse Response --- 
        let evaluation: OpportunityEvaluation = serde_json::from_str(&response_string)
            .map_err(|e| {
                error!(error = %e, raw_response = %response_string, "Failed to parse JSON response from RIG agent");
                SandoError::rig(RigErrorKind::InvalidResponse, format!("Failed to parse agent JSON response: {}", e))
            })?;

        if let Some(ref err_msg) = evaluation.error {
            warn!(agent_error = %err_msg, "RIG agent reported an error in its evaluation.");
        }

        info!(evaluation = ?evaluation, "Successfully parsed agent evaluation.");
        // Log the reasoning explicitly for monitoring/debugging
        info!(reasoning = %evaluation.reasoning, agent_suggestion = %evaluation.suggested_action, "Agent evaluation reasoning");
        Ok(evaluation)
    }

    /// Fallback evaluation logic using simple rules when AI confidence is low.
    pub fn evaluate_opportunity_rule_based(
        &self,
        raw_opportunity: &RawOpportunityData,
        market_context: &MarketContext,
    ) -> Result<OpportunityEvaluation> {
        info!(opportunity_hash = %raw_opportunity.transaction_hash, "Evaluating opportunity with RULE-BASED fallback...");
        // --- Implement Simple Rule-Based Logic --- 
        // Example: Basic profitability check (needs refinement)
        let estimated_cost_usd = 0.1; // Placeholder for gas/fees
        let potential_profit = (raw_opportunity.output_amount * market_context.output_token_price_usd) 
                               - (raw_opportunity.input_amount * market_context.input_token_price_usd)
                               - estimated_cost_usd;
        
        let is_viable = potential_profit > 1.0; // Arbitrary threshold
        let confidence = if is_viable { 0.9 } else { 0.1 }; // High confidence in simple rule
        let suggestion = if is_viable { "Execute Arbitrage" } else { "Decline" };

        let evaluation = OpportunityEvaluation {
            is_viable,
            confidence_score: confidence, // Rule based has fixed confidence
            estimated_profit_usd: potential_profit,
            reasoning: format!("Rule-based check: Estimated profit {:.2} USD. Threshold: 1.0 USD.", potential_profit),
            suggested_action: suggestion.to_string(),
            error: None,
        };

        warn!(evaluation = ?evaluation, "Used rule-based fallback evaluation.");
        Ok(evaluation)
    }

    // TODO: Add functions for market data integration (Task 6.2)
    // TODO: Add functions for decision reasoning/fallback (Task 6.3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    // Note: These tests require setting ANTHROPIC_API_KEY environment variable
    // and may incur costs.
    
    #[tokio::test]
    #[ignore] // Ignore by default to avoid running without API key / cost
    async fn test_rig_agent_from_env_and_prompt() {
        // Ensure API key is set (replace with your actual key for testing)
        // !!! Do not commit your real API key !!!
        let api_key = env::var("ANTHROPIC_API_KEY");
        if api_key.is_err() {
            println!("Skipping test_rig_agent_from_env_and_prompt: ANTHROPIC_API_KEY not set.");
            return;
            // env::set_var("ANTHROPIC_API_KEY", "YOUR_TEST_KEY_HERE"); 
        }

        let agent_result = RigAgent::from_env();
        assert!(agent_result.is_ok(), "Failed to create agent: {:?}", agent_result.err());
        let agent = agent_result.unwrap();

        let raw_opportunity = RawOpportunityData {
            source_dex: "Orca".to_string(),
            transaction_hash: "test_tx_hash".to_string(),
            input_token: "SOL".to_string(),
            output_token: "USDC".to_string(),
            input_amount: 100.0,
            output_amount: 10000.0,
        };

        let market_context = MarketContext {
            input_token_price_usd: 100.0,
            output_token_price_usd: 1.0,
            pool_liquidity_usd: 1_000_000.0,
            recent_volatility_percent: 5.0,
            market_token_prices: HashMap::new(),
            market_token_volatility: HashMap::new(),
            market_token_price_changes: HashMap::new(),
            market_token_forecasts: HashMap::new(),
            average_market_price: 0.0,
            average_market_volatility: 0.0,
            average_market_price_change: 0.0,
        };

        let response_result = agent.evaluate_opportunity(&raw_opportunity, &market_context).await;
        assert!(response_result.is_ok());

        let response = response_result.unwrap();
        println!("Test prompt response: {:?}", response);
        assert!(!response.reasoning.is_empty());
    }

    #[tokio::test]
    #[ignore] // Still requires API key
    async fn test_evaluate_opportunity_format() {
        // This test checks if the agent can be prompted with context and if the response
        // can be *structurally* parsed, not the logical correctness of the evaluation.
        let api_key = env::var("ANTHROPIC_API_KEY");
        if api_key.is_err() {
            println!("Skipping test_evaluate_opportunity_format: ANTHROPIC_API_KEY not set.");
            return;
        }

        let agent = RigAgent::from_env().expect("Failed to create agent");

        let opportunity_data = RawOpportunityData {
            source_dex: "ExampleDEX".to_string(),
            transaction_hash: "dummy_hash_123".to_string(),
            input_token: "USDC".to_string(),
            output_token: "SOL".to_string(),
            input_amount: 1000.0,
            output_amount: 5.0, // Example data
        };

        let market_data = MarketContext {
            input_token_price_usd: 1.0,
            output_token_price_usd: 160.0,
            pool_liquidity_usd: 500000.0,
            recent_volatility_percent: 1.5,
            market_token_prices: HashMap::new(),
            market_token_volatility: HashMap::new(),
            market_token_price_changes: HashMap::new(),
            market_token_forecasts: HashMap::new(),
            average_market_price: 0.0,
            average_market_volatility: 0.0,
            average_market_price_change: 0.0,
        };

        let result = agent.evaluate_opportunity(&opportunity_data, &market_data).await;

        match result {
            Ok(evaluation) => {
                println!("Parsed Evaluation: {:?}", evaluation);
                // Basic structural checks
                assert!(evaluation.confidence_score >= 0.0 && evaluation.confidence_score <= 1.0);
                assert!(!evaluation.reasoning.is_empty());
                assert!(!evaluation.suggested_action.is_empty());
            }
            Err(e) => {
                // If it fails, likely due to LLM not returning valid JSON or API error
                panic!("Evaluation failed: {:?}. Check LLM response or API key.", e);
            }
        }
    }

    #[test]
    fn test_evaluate_opportunity_rule_based() {
         let agent = RigAgent::from_env().expect("Failed to create agent for rule test");

        let opportunity_data_profit = RawOpportunityData {
            source_dex: "ExampleDEX".to_string(),
            transaction_hash: "profit_hash_456".to_string(),
            input_token: "USDC".to_string(),
            output_token: "SOL".to_string(),
            input_amount: 100.0,
            output_amount: 0.65, // 0.65 * 160 = 104 -> profit > 1
        };
         let opportunity_data_loss = RawOpportunityData {
            source_dex: "ExampleDEX".to_string(),
            transaction_hash: "loss_hash_789".to_string(),
            input_token: "USDC".to_string(),
            output_token: "SOL".to_string(),
            input_amount: 100.0,
            output_amount: 0.60, // 0.6 * 160 = 96 -> profit < 1
        };

        let market_data = MarketContext {
            input_token_price_usd: 1.0,
            output_token_price_usd: 160.0,
            pool_liquidity_usd: 500000.0,
            recent_volatility_percent: 1.5,
            market_token_prices: HashMap::new(),
            market_token_volatility: HashMap::new(),
            market_token_price_changes: HashMap::new(),
            market_token_forecasts: HashMap::new(),
            average_market_price: 0.0,
            average_market_volatility: 0.0,
            average_market_price_change: 0.0,
        };

        let eval_profit = agent.evaluate_opportunity_rule_based(&opportunity_data_profit, &market_data);
        assert!(eval_profit.is_ok());
        let eval_profit = eval_profit.unwrap();
        assert!(eval_profit.is_viable);
        assert!(eval_profit.estimated_profit_usd > 1.0);
        assert_eq!(eval_profit.suggested_action, "Execute Arbitrage");

        let eval_loss = agent.evaluate_opportunity_rule_based(&opportunity_data_loss, &market_data);
        assert!(eval_loss.is_ok());
        let eval_loss = eval_loss.unwrap();
        assert!(!eval_loss.is_viable);
        assert!(eval_loss.estimated_profit_usd < 1.0);
        assert_eq!(eval_loss.suggested_action, "Decline");
    }
} 