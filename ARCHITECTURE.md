# SandoSeer Architecture - Component Interaction and Data Flow

This document describes the core architecture of the SandoSeer system, particularly focusing on the component interaction and data flow as specified in the Product Requirements Document (PRD).

## System Overview

SandoSeer is an autonomous MEV (Maximal Extractable Value) detection and execution system for the Solana blockchain. It leverages AI-powered decision making through the `rig` framework and real-time transaction monitoring via the `listen` engine to identify and capitalize on MEV opportunities including sandwich trading, arbitrage, and token sniping.

## Core Components

The system is composed of the following key components, each with specific responsibilities:

1. **Listen Bot**: Monitors Solana DEX transactions and mempool in real-time
2. **Market Data Collector**: Gathers price and liquidity information
3. **RIG Agent (LLM)**: Provides AI-driven forecasting and decision-making
4. **Opportunity Evaluator**: Calculates risk-adjusted profitability of MEV opportunities
5. **Transaction Executor**: Executes profitable trades based on identified opportunities
6. **Monitoring & Feedback**: Tracks performance and provides alerts

## Component Interaction

The components interact in a well-defined flow to process transaction data and execute MEV opportunities:

```
┌────────────────────────┐
│ RIG AGENT (LLM)        │
│ "Sees the future"      │
└────────┬───────────────┘
         │
         ▼
┌────────────────────┐ ┌─────────────┐ ┌───────────────────────┐
│ Social + Token     │ │ Market Data │ │ Mempool / TX Signals  │
│ Sentiment Scanner  │ │ (Orca etc)  │ │ (via Listen)          │
└────────┬───────────┘ └─────┬───────┘ └──────────────┬────────┘
         ▼                   ▼                        ▼
┌────────────────────────────────────────────────────────────┐
│ Opportunity Evaluator                                      │
│ - Predicts MEV chance                                      │
│ - Calculates confidence + profitability                    │
│ - Adjusts risk parameters dynamically                      │
└────────────┬──────────────────────────────────────────────┘
             ▼
┌───────────────┐
│ LISTEN BOT    │ <-- Executes trades (swap, arb, sandwich)
└───────────────┘
```

### Interaction Details

1. **Listen Bot → Opportunity Evaluator**: The Listen Bot streams transaction data from Solana DEXes and forwards potential MEV opportunities to the Opportunity Evaluator.

2. **Market Data Collector → Opportunity Evaluator**: The Market Data Collector provides real-time price and liquidity information to enrich transaction data for evaluation.

3. **Market Data + Transaction Data → RIG Agent**: The combined market and transaction data is sent to the RIG Agent for AI-powered analysis.

4. **RIG Agent → Opportunity Evaluator**: The RIG Agent provides opportunity scores, confidence ratings, and strategy recommendations to the Opportunity Evaluator.

5. **Opportunity Evaluator → Decision Making**: The Opportunity Evaluator calculates final scores, performs risk assessment, and makes execution decisions.

6. **Decision Making → Transaction Executor**: Profitable opportunities are forwarded to the Transaction Executor for implementation.

7. **Transaction Executor → Monitoring**: Execution results are logged and monitored for performance tracking and feedback.

## Data Flow

The data flows through the system as follows:

1. **Incoming transaction data → Listen Bot**
   - Transaction data is captured from the Solana blockchain
   - Basic filtering and preprocessing is applied
   - Potential MEV opportunities are identified

2. **Transaction data + Market data + Sentiment data → RIG Agent**
   - Transaction details are combined with market conditions
   - Sentiment analysis is added for context
   - The combined data is analyzed by the AI agent

3. **RIG Agent decision + Transaction data → Opportunity Evaluator**
   - AI insights are merged with transaction data
   - Opportunity scoring algorithms are applied
   - Risk assessment is performed

4. **Opportunity score + Transaction details → Decision Maker**
   - Scores are compared against thresholds
   - Risk parameters are considered
   - Go/no-go decisions are made

5. **Trade decision → Transaction Executor**
   - Transaction construction is optimized
   - Safety checks are performed
   - Transactions are executed on-chain

6. **Execution results → Monitoring & Logging**
   - Transaction outcomes are recorded
   - Performance metrics are updated
   - Notifications are sent as needed

## Implementation Status

The current implementation has established the core components and their interaction patterns:

- **ListenBot**: Implemented with integration to the `listen-core` engine
- **RigAgent**: Integrated with the Anthropic Claude API through the `rig` framework
- **OpportunityEvaluator**: Implemented with scoring algorithms and risk assessment
- **MarketDataCollector**: Basic implementation capturing price and liquidity data
- **ExecutionService**: Implemented with support for multiple MEV strategies
- **SandoEngine**: Orchestration layer that manages the data flow between components

## Future Enhancements

The following enhancements are planned to improve the component interaction and data flow:

1. **Enhance RIG Agent Integration**: Deeper integration of AI decision-making throughout the evaluation pipeline
2. **Expand Market Data Sources**: Add more market data providers and sentiment analysis
3. **Improve Strategy Selection**: Smarter selection between different MEV strategies based on market conditions
4. **Implement Feedback Loop**: Use execution results to improve future opportunity evaluation
5. **Add Circuit Breakers**: Implement safeguards to protect against adverse market conditions

## Conclusion

The SandoSeer architecture is designed to maximize MEV extraction while managing risk through a well-defined component interaction and data flow. The system combines real-time transaction monitoring, AI-powered decision making, and efficient execution strategies to capture value from the Solana DeFi ecosystem. 