use listen_engine::{
    engine::Engine,
    filters::ProgramFilter,
    pipeline::{Pipeline, PipelineBuilder},
    redis::subscriber::PriceUpdate,
};

use crate::error::Result;
use tokio::sync::mpsc;
use tracing::{info, error, warn};
use std::sync::Arc;

mod config;
mod types;
mod dex;
mod transaction;
mod stream;

pub use config::ListenBotConfig;
pub use types::*;
use dex::{DexParserFactory, DexType, DexSwap};
pub use transaction::{DexTransaction, DexTransactionParser, TokenAmount, TransactionMonitor};
pub use stream::{StreamConfig, TransactionStream};

/// Main ListenBot struct that manages transaction monitoring
pub struct ListenBot {
    /// The underlying listen-engine instance
    engine: Arc<Engine>,
    /// Price update receiver
    price_rx: mpsc::Receiver<PriceUpdate>,
    /// Configuration
    config: ListenBotConfig,
    /// Transaction event sender
    event_tx: mpsc::Sender<TransactionEvent>,
    /// Swap handler
    swap_handler: TokenSwapHandler,
    /// Command sender
    cmd_tx: mpsc::Sender<()>,
}

impl ListenBot {
    /// Creates a new ListenBot instance
    pub async fn new(config: ListenBotConfig) -> Result<mpsc::Sender<()>> {
        let (cmd_tx, cmd_rx) = mpsc::channel(100);
        let (price_tx, price_rx) = mpsc::channel(100);
        let (engine, price_rx) = Engine::from_env().await?;
        let engine = Arc::new(engine);
        let (event_tx, event_rx) = mpsc::channel(1000);

        // Create swap handler with Redis store and message queue from engine
        let swap_handler = TokenSwapHandler::new(
            engine.redis.clone(),
            engine.redis_sub.clone(),
            engine.privy.clone(),
        );

        Ok(cmd_tx.clone())
    }

    /// Starts the ListenBot
    pub async fn start(&mut self) -> Result<mpsc::Sender<()>> {
        info!("Starting ListenBot");

        let (cmd_tx, mut cmd_rx) = mpsc::channel(1);
        let pipeline = self.create_monitoring_pipeline().await?;
        
        // Save pipeline to Redis
        let mut pipeline_hash = String::new();
        self.engine.save_pipeline(&pipeline, &mut pipeline_hash).await?;
        
        // Start the engine
        let engine_clone = self.engine.clone();
        tokio::spawn(async move {
            if let Err(e) = Engine::run(
                engine_clone,
                cmd_rx,
                mpsc::channel(100).1, // Dummy command receiver
            ).await {
                error!("Engine error: {}", e);
            }
        });

        Ok(cmd_tx)
    }

    /// Creates a monitoring pipeline based on configuration
    async fn create_monitoring_pipeline(&self) -> Result<Pipeline> {
        let mut builder = PipelineBuilder::new();
        let event_tx = self.event_tx.clone();
        let swap_handler = self.swap_handler.clone();

        // Add program filters if specified
        if !self.config.filter.program_ids.is_empty() {
            builder = builder.add_filter(ProgramFilter::new(self.config.filter.program_ids.clone()));
        }

        // Add transaction processor
        builder = builder.add_processor(move |tx| {
            let event_tx = event_tx.clone();
            let swap_handler = swap_handler.clone();
            
            Box::pin(async move {
                let timestamp = chrono::Utc::now().timestamp();
                
                // Try to create a basic transaction event
                if let Some(mut event) = TransactionEvent::from_confirmed_tx(tx.clone(), timestamp) {
                    // Process as a DEX swap using the built-in handler
                    for dex in [Dex::RaydiumAmmV4, Dex::RaydiumClmm, Dex::Whirlpools] {
                        if let Some(swap_info) = swap_handler.process_swap(
                            &tx,
                            dex,
                            &event.program_ids,
                            &event.token_mints,
                        ) {
                            // Update event with swap info
                            event.tx_type = TransactionType::Swap;
                            event.amount = swap_info.input_amount;
                            event.metadata.extra.insert("dex".to_string(), format!("{:?}", dex));
                            event.metadata.extra.insert("output_amount".to_string(), swap_info.output_amount.to_string());
                            
                            // Send event
                            if let Err(e) = event_tx.try_send(event) {
                                error!("Failed to send transaction event: {}", e);
                            }
                            return;
                        }
                    }

                    // Try Jupiter if no other DEX matched
                    if let Ok(quote) = Jupiter::fetch_quote(
                        &event.token_mints[0].to_string(),
                        &event.token_mints[1].to_string(),
                        event.amount,
                    ).await {
                        event.tx_type = TransactionType::Swap;
                        event.metadata.extra.insert("dex".to_string(), "Jupiter".to_string());
                        event.metadata.extra.insert("output_amount".to_string(), quote.out_amount);
                        
                        if let Err(e) = event_tx.try_send(event) {
                            error!("Failed to send transaction event: {}", e);
                        }
                    }
                }
            })
        });

        // Add size filter if specified
        if self.config.filter.min_size > 0 {
            builder = builder.add_filter(move |tx| {
                tx.meta
                    .as_ref()
                    .map(|meta| meta.fee >= self.config.filter.min_size)
                    .unwrap_or(false)
            });
        }

        // Add success/failure filter
        if !self.config.filter.include_failed {
            builder = builder.add_filter(|tx| {
                tx.meta
                    .as_ref()
                    .map(|meta| meta.err.is_none())
                    .unwrap_or(false)
            });
        }

        Ok(builder.build())
    }

    /// Stops the ListenBot
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping ListenBot");
        self.engine.shutdown().await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;
    use std::time::Duration;

    #[tokio::test]
    async fn test_listenbot_lifecycle() -> Result<()> {
        let config = ListenBotConfig::default();
        let (mut bot, _rx) = ListenBot::new(config).await?;

        // Start the bot
        let cmd_tx = bot.start().await?;
        sleep(Duration::from_secs(1)).await;

        // Stop the bot
        drop(cmd_tx);
        bot.stop().await?;

        Ok(())
    }
} 