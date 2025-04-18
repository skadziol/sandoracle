use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, Instant};
use tokio::sync::RwLock;
use crate::error::{Result, SandoError};
use crate::jupiter_client::Jupiter;
use tracing::{info, error, warn, debug};
use futures::future;
use tokio::time::sleep;
use std::cmp::min;
use solana_client::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use std::str::FromStr;

/// Enhanced market data with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub price: f64,
    pub market_cap: f64,
    pub volume_24h: f64,
    pub price_change_24h: f64,
    pub liquidity: f64,
    pub volatility: f64,
    pub last_update: u64,
    
    // TTL tracking (not serialized)
    #[serde(skip)]
    pub expires_at: Option<SystemTime>,
    
    // Data source tracking
    #[serde(default)]
    pub source: String,
    
    // Confidence score (0.0-1.0)
    #[serde(default = "default_confidence")]
    pub confidence: f64,
}

fn default_confidence() -> f64 {
    0.95 // Default high confidence
}

/// Cache entry with TTL
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    data: T,
    expires_at: SystemTime,
}

impl<T> CacheEntry<T> {
    fn new(data: T, ttl_seconds: u64) -> Self {
        let expires_at = SystemTime::now() + Duration::from_secs(ttl_seconds);
        Self { data, expires_at }
    }
    
    fn is_expired(&self) -> bool {
        SystemTime::now() > self.expires_at
    }
}

/// Collector for market data from various sources
#[derive(Clone)]
pub struct MarketDataCollector {
    /// Price cache with token_mint -> price mapping
    price_cache: Arc<RwLock<HashMap<String, CacheEntry<f64>>>>,
    
    /// Historical data with token_mint -> [prices] mapping
    historical_data: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    
    /// Jupiter client for API access
    jupiter_client: Option<Jupiter>,
    
    /// Default TTL for price cache in seconds
    default_ttl: u64,
    
    /// Full market data cache with token_mint -> MarketData
    market_data_cache: Arc<RwLock<HashMap<String, CacheEntry<MarketData>>>>,
    
    /// Last successful batch update timestamp
    last_batch_update: Arc<RwLock<Option<Instant>>>,
    
    /// Maximum retry attempts for API calls
    max_retries: u32,
    
    /// List of supported token mints
    supported_tokens: Arc<RwLock<HashMap<String, bool>>>,
    
    // Multi-source price provider (commented out)
    // multi_source_provider: Option<MultiSourceProvider>,
}

impl MarketDataCollector {
    /// Create a new market data collector
    pub fn new(
        price_cache: Arc<RwLock<HashMap<String, f64>>>,
        historical_data: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    ) -> Self {
        // Convert existing price cache to new format with TTL
        let new_cache = Arc::new(RwLock::new(HashMap::new()));
        let market_data_cache = Arc::new(RwLock::new(HashMap::new()));
        let last_batch_update = Arc::new(RwLock::new(None));
        let supported_tokens = Arc::new(RwLock::new(HashMap::new()));
        
        let collector = Self {
            price_cache: new_cache,
            historical_data,
            jupiter_client: None,
            default_ttl: 300, // 5 minutes default TTL
            market_data_cache,
            last_batch_update,
            max_retries: 3,
            supported_tokens,
            // multi_source_provider: None,
        };
        
        // Spawn a task to migrate old cache to new format
        let collector_clone = collector.clone();
        let old_cache = price_cache.clone();
        tokio::spawn(async move {
            let old_data = old_cache.read().await;
            let mut new_data = collector_clone.price_cache.write().await;
            
            for (token, &price) in old_data.iter() {
                new_data.insert(
                    token.clone(),
                    CacheEntry::new(price, collector_clone.default_ttl),
                );
            }
            
            debug!("Migrated {} prices from old cache to new TTL-based cache", old_data.len());
        });
        
        collector
    }

    /// Initialize with Jupiter API for real-time price data
    pub fn with_jupiter(mut self, jupiter_api_url: &str) -> Result<Self> {
        // Initialize Jupiter client
        self.jupiter_client = Some(Jupiter::new(jupiter_api_url));
        
        // Remove all price provider related code
        // let jupiter_provider = match JupiterProvider::new(jupiter_api_url) {
        //     Ok(provider) => provider,
        //     Err(e) => {
        //         error!("Failed to initialize Jupiter price provider: {}", e);
        //         return Err(e);
        //     }
        // };
        // 
        // // Initialize multi-source provider with Jupiter
        // let providers: Vec<Box<dyn PriceProvider>> = vec![
        //     Box::new(jupiter_provider),
        // ];
        
        info!("Initialized Jupiter client for real-time market data");
        Ok(self)
    }
    
    /// Set the default TTL for cached data
    pub fn with_default_ttl(mut self, ttl_seconds: u64) -> Self {
        self.default_ttl = ttl_seconds;
        self
    }
    
    /// Set the maximum retry attempts for API calls
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Update market data for a token
    pub async fn update_market_data(&self, token: &str, data: MarketData) -> Result<()> {
        // Save the price before moving the data
        let price = data.price;
        
        // Update price in cache with TTL
        {
            let mut cache = self.price_cache.write().await;
            cache.insert(token.to_string(), CacheEntry::new(price, self.default_ttl));
        }
        
        // Update market data cache
        {
            let mut cache = self.market_data_cache.write().await;
            cache.insert(token.to_string(), CacheEntry::new(data, self.default_ttl));
        }
        
        // Update historical data (keep last 24 data points)
        {
            let mut history = self.historical_data.write().await;
            let token_history = history.entry(token.to_string()).or_insert_with(Vec::new);
            token_history.push(price);
            
            // Keep only the last 24 data points
            if token_history.len() > 24 {
                token_history.remove(0);
            }
        }
        
        Ok(())
    }

    /// Fetch real market data from Jupiter API with retries and backoff
    pub async fn fetch_real_market_data(&self, token: &str, reference_token: &str) -> Result<Option<f64>> {
        // Check cache first
        {
            let cache = self.price_cache.read().await;
            if let Some(entry) = cache.get(token) {
                if !entry.is_expired() {
                    debug!(token=%token, "Using cached price data");
                    return Ok(Some(entry.data));
                }
            }
        }
        
        // If token and reference token are the same, the price is always 1:1
        if token == reference_token {
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
                
            // Update cache
            {
                let mut cache = self.price_cache.write().await;
                cache.insert(token.to_string(), CacheEntry::new(1.0, 86400)); // 24 hour TTL for 1:1 rates
            }
            
            // Create and update full market data
            let market_data = MarketData {
                price: 1.0,
                market_cap: 0.0, // Unknown
                volume_24h: 0.0, // Unknown
                price_change_24h: 0.0, // No change for same token
                liquidity: 0.0, // Unknown
                volatility: 0.0, // No volatility for same token
                last_update: now,
                expires_at: Some(SystemTime::now() + Duration::from_secs(86400)),
                source: "identity".to_string(),
                confidence: 1.0,
            };
            
            self.update_market_data(token, market_data).await?;
            
            info!(token=%token, "Token and reference token are identical, returning 1.0 as price");
            return Ok(Some(1.0));
        }
        
        // Check if token is marked as unsupported to avoid retries
        {
            let supported = self.supported_tokens.read().await;
            if let Some(&is_supported) = supported.get(token) {
                if !is_supported {
                    debug!(token=%token, "Token previously marked as unsupported, skipping");
                    return Ok(None);
                }
            }
        }
        
        // Try to get price from multi-source provider first if available
        // if let Some(provider) = &self.multi_source_provider {
        //     debug!(token=%token, "Attempting to get price from multi-source provider");
        //     
        //     // First try to get all prices and aggregate them
        //     let prices = provider.get_prices_from_all_sources(token, reference_token).await?;
        //     
        //     if !prices.is_empty() {
        //         debug!(token=%token, sources=%prices.len(), "Got prices from multiple sources");
        //         
        //         // Aggregate the prices
        //         if let Some(aggregate) = provider.aggregate_prices(&prices) {
        //             // Update cache with the aggregated price
        //             {
        //                 let mut cache = self.price_cache.write().await;
        //                 cache.insert(
        //                     token.to_string(), 
        //                     CacheEntry::new(aggregate.price, self.default_ttl)
        //                 );
        //             }
        //             
        //             // Create market data from the aggregate
        //             let market_data = MarketData {
        //                 price: aggregate.price,
        //                 market_cap: 0.0, // Unknown from price alone
        //                 volume_24h: 0.0, // Unknown from price alone
        //                 price_change_24h: 0.0, // Need historical data
        //                 liquidity: 0.0, // Unknown from price alone
        //                 volatility: 0.0, // Need historical data
        //                 last_update: aggregate.timestamp,
        //                 expires_at: Some(SystemTime::now() + Duration::from_secs(self.default_ttl)),
        //                 source: format!("aggregate-{}", prices.len()),
        //                 confidence: aggregate.confidence,
        //             };
        //             
        //             // Update market data cache
        //             self.update_market_data(token, market_data).await?;
        //             
        //             // Mark as supported
        //             {
        //                 let mut supported = self.supported_tokens.write().await;
        //                 supported.insert(token.to_string(), true);
        //             }
        //             
        //             return Ok(Some(aggregate.price));
        //         }
        //     }
        //     
        //     // If aggregation failed, try to get the best price
        //     if let Ok(Some(best_price)) = provider.get_best_price(token, reference_token).await {
        //         debug!(token=%token, source=?best_price.source, "Using best price from available sources");
        //         
        //         // Update cache with the best price
        //         {
        //             let mut cache = self.price_cache.write().await;
        //             cache.insert(
        //                 token.to_string(), 
        //                 CacheEntry::new(best_price.price, self.default_ttl)
        //             );
        //         }
        //         
        //         // Create market data from the best price
        //         let market_data = MarketData {
        //             price: best_price.price,
        //             market_cap: 0.0, // Unknown from price alone
        //             volume_24h: 0.0, // Unknown from price alone
        //             price_change_24h: 0.0, // Need historical data
        //             liquidity: 0.0, // Unknown from price alone
        //             volatility: 0.0, // Need historical data
        //             last_update: best_price.timestamp,
        //             expires_at: Some(SystemTime::now() + Duration::from_secs(self.default_ttl)),
        //             source: format!("{:?}", best_price.source),
        //             confidence: best_price.confidence,
        //         };
        //         
        //         // Update market data cache
        //         self.update_market_data(token, market_data).await?;
        //         
        //         // Mark as supported
        //         {
        //             let mut supported = self.supported_tokens.write().await;
        //             supported.insert(token.to_string(), true);
        //         }
        //         
        //         return Ok(Some(best_price.price));
        //     }
        // }
        
        // Fall back to direct Jupiter client if multi-source fails or is not available
        if let Some(jupiter) = &self.jupiter_client {
            // Try with exponential backoff
            let mut retry_count = 0;
            let mut backoff_ms = 500; // Start with 500ms
            
            while retry_count <= self.max_retries {
                match jupiter.get_price(token, reference_token).await {
                    Ok(price) => {
                        // Update cache with the fetched price
                        {
                            let mut cache = self.price_cache.write().await;
                            cache.insert(
                                token.to_string(), 
                                CacheEntry::new(price, self.default_ttl)
                            );
                        }
                        
                        // Create simple market data from price
                        let now = SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                            
                        let market_data = MarketData {
                            price,
                            market_cap: 0.0, // Unknown from price alone
                            volume_24h: 0.0, // Unknown from price alone
                            price_change_24h: 0.0, // Need historical data
                            liquidity: 0.0, // Unknown from price alone
                            volatility: 0.0, // Need historical data
                            last_update: now,
                            expires_at: Some(SystemTime::now() + Duration::from_secs(self.default_ttl)),
                            source: "jupiter".to_string(),
                            confidence: 0.95,
                        };
                        
                        // Update market data cache
                        {
                            let mut cache = self.market_data_cache.write().await;
                            cache.insert(
                                token.to_string(), 
                                CacheEntry::new(market_data, self.default_ttl)
                            );
                        }
                        
                        // Add to supported tokens list
                        {
                            let mut supported = self.supported_tokens.write().await;
                            supported.insert(token.to_string(), true);
                        }
                        
                        return Ok(Some(price));
                    },
                    Err(e) => {
                        warn!(
                            token=%token,
                            error=%e,
                            retry_count=%retry_count,
                            backoff_ms=%backoff_ms,
                            "Jupiter price fetch failed, retrying with backoff"
                        );
                        
                        // Wait before retry with exponential backoff
                        sleep(Duration::from_millis(backoff_ms)).await;
                        retry_count += 1;
                        backoff_ms = min(backoff_ms * 2, 8000); // Cap at 8 seconds
                    }
                }
            }
        }
        
        // If multi-source provider and Jupiter client both failed, try to get from cache even if expired
        let cache = self.price_cache.read().await;
        if let Some(entry) = cache.get(token) {
            warn!(token=%token, "Using expired cache data due to API failure");
            return Ok(Some(entry.data));
        }
        
        Ok(None)
    }

    /// Fetch market data for multiple tokens in parallel with intelligent batching
    pub async fn fetch_batch_market_data(
        &self, 
        token_mints: &HashMap<&str, &str>, 
        reference_token: &str
    ) -> Result<HashMap<String, Option<f64>>> {
        let start_time = Instant::now();
        let mut results = HashMap::new();
        
        // Limit batches to process 20 tokens at a time
        const BATCH_SIZE: usize = 20;
        
        // Mark batch update start time
        {
            let mut last_update = self.last_batch_update.write().await;
            *last_update = Some(Instant::now());
        }
        
        // First, check if we can use cached values for any tokens
        let tokens_to_fetch = {
            let cache = self.price_cache.read().await;
            
            // Store cached results and collect tokens that need fetching
            let mut to_fetch = Vec::new();
            
            for (&token_name, &token_mint) in token_mints.iter() {
                // Special case: token is the reference token
                if token_mint == reference_token {
                    results.insert(token_name.to_string(), Some(1.0));
                    continue;
                }
                
                // Check cache
                if let Some(entry) = cache.get(token_mint) {
                    if !entry.is_expired() {
                        results.insert(token_name.to_string(), Some(entry.data));
                        continue;
                    }
                }
                
                // Need to fetch this token
                to_fetch.push((token_name, token_mint));
            }
            
            to_fetch
        };
        
        // If nothing to fetch, return early
        if tokens_to_fetch.is_empty() {
            debug!("All tokens found in cache, no need for API calls");
            return Ok(results);
        }
        
        info!(
            to_fetch_count=%tokens_to_fetch.len(),
            cached_count=%results.len(),
            "Starting batch market data fetch"
        );
        
        // Process tokens in batches
        for chunk in tokens_to_fetch.chunks(BATCH_SIZE) {
            let ref_token = reference_token.to_string();
            
            // Create futures for parallel fetching
            let futures = chunk.iter().map(|&(token_name, token_mint)| {
                let self_clone = self.clone();
                let token_name = token_name.to_string();
                let token_mint = token_mint.to_string();
                let ref_token = ref_token.clone();
                
                async move {
                    match self_clone.fetch_real_market_data(&token_mint, &ref_token).await {
                        Ok(price) => (token_name, price),
                        Err(e) => {
                            error!(token=%token_mint, error=%e, "Error in batch fetch");
                            (token_name, None)
                        }
                    }
                }
            });
            
            // Execute batch in parallel
            let batch_results = future::join_all(futures).await;
            
            // Collect batch results
            for (token_name, price) in batch_results {
                results.insert(token_name, price);
            }
            
            // Brief pause between batches to avoid rate limits
            if chunk.len() == BATCH_SIZE {
                sleep(Duration::from_millis(100)).await;
            }
        }
        
        let elapsed = start_time.elapsed();
        info!(
            fetched_count=%tokens_to_fetch.len(),
            total_count=%token_mints.len(),
            duration_ms=%elapsed.as_millis(),
            "Completed batch market data fetch"
        );
        
        Ok(results)
    }
    
    /// Get the historical price data
    pub async fn get_historical_data(&self) -> HashMap<String, Vec<f64>> {
        let history = self.historical_data.read().await;
        history.clone()
    }
    
    /// Get historical data for a specific token
    pub async fn get_token_history(&self, token: &str) -> Option<Vec<f64>> {
        let history = self.historical_data.read().await;
        history.get(token).cloned()
    }

    /// Get reference to the Jupiter client if available
    pub fn get_jupiter_client(&self) -> Option<&Jupiter> {
        self.jupiter_client.as_ref()
    }
    
    /// Check if a token is supported
    pub async fn is_token_supported(&self, token: &str) -> bool {
        // First check the cache
        {
            let supported = self.supported_tokens.read().await;
            if let Some(&value) = supported.get(token) {
                return value;
            }
        }
        
        // Try fetching the price to determine support
        match self.fetch_real_market_data(token, "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v").await {
            Ok(Some(_)) => {
                // Update supported token cache
                let mut supported = self.supported_tokens.write().await;
                supported.insert(token.to_string(), true);
                true
            },
            _ => {
                // Mark as unsupported
                let mut supported = self.supported_tokens.write().await;
                supported.insert(token.to_string(), false);
                false
            }
        }
    }
    
    /// Get full market data for a token
    pub async fn get_market_data(&self, token: &str, reference_token: &str) -> Result<Option<MarketData>> {
        // Check full data cache first
        {
            let cache = self.market_data_cache.read().await;
            if let Some(entry) = cache.get(token) {
                if !entry.is_expired() {
                    return Ok(Some(entry.data.clone()));
                }
            }
        }
        
        // If it's not in cache or expired, build it from price and history
        if let Ok(Some(price)) = self.fetch_real_market_data(token, reference_token).await {
            let history = self.get_token_history(token).await.unwrap_or_default();
            
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            // Calculate price change if we have history
            let price_change = if history.len() >= 2 {
                let oldest = history.first().unwrap_or(&price);
                ((price - oldest) / oldest) * 100.0
            } else {
                0.0
            };
            
            // Calculate volatility if we have enough history
            let volatility = if history.len() >= 3 {
                // Simple volatility calculation
                history.windows(2)
                    .map(|w| ((w[1] - w[0]) / w[0]).abs())
                    .sum::<f64>() / (history.len() - 1) as f64
            } else {
                0.01 // Default low volatility
            };
            
            let market_data = MarketData {
                price,
                market_cap: 0.0,      // Would need additional API call
                volume_24h: 0.0,      // Would need additional API call
                price_change_24h: price_change,
                liquidity: 0.0,       // Would need additional API call
                volatility,
                last_update: now,
                expires_at: Some(SystemTime::now() + Duration::from_secs(self.default_ttl)),
                source: "jupiter".to_string(),
                confidence: 0.95,
            };
            
            // Update the cache
            {
                let mut cache = self.market_data_cache.write().await;
                cache.insert(token.to_string(), CacheEntry::new(market_data.clone(), self.default_ttl));
            }
            
            return Ok(Some(market_data));
        }
        
        Ok(None)
    }
    
    /// Get a simple price forecast for a token based on historical data
    pub async fn get_price_forecast(&self, token: &str, minutes_ahead: u64) -> Result<Option<f64>> {
        // Needs at least 3 data points for a meaningful trend
        let history = match self.get_token_history(token).await {
            Some(h) if h.len() >= 3 => h,
            _ => return Ok(None),
        };
        
        // Simple linear regression for forecasting
        let n = history.len();
        let x_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y_vals = history;
        
        let sum_x: f64 = x_vals.iter().sum();
        let sum_y: f64 = y_vals.iter().sum();
        let sum_xy: f64 = x_vals.iter().zip(y_vals.iter())
            .map(|(&x, &y)| x * y)
            .sum();
        let sum_xx: f64 = x_vals.iter().map(|&x| x * x).sum();
        
        let n_f64 = n as f64;
        
        // Calculate slope (m) and y-intercept (b)
        let slope = (n_f64 * sum_xy - sum_x * sum_y) / (n_f64 * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n_f64;
        
        // Calculate forecast point (x = n + minutes ahead)
        let minutes_per_sample = 5.0; // Assuming 5 minute intervals in our history
        let x_forecast = n as f64 + (minutes_ahead as f64 / minutes_per_sample);
        let forecast = slope * x_forecast + intercept;
        
        Ok(Some(forecast))
    }

    /// Add the Birdeye provider (stub implementation)
    pub fn with_birdeye(mut self, _api_key: String) -> Result<Self> {
        // Just return self - Birdeye provider is not implemented
        info!("Birdeye integration not yet implemented");
        Ok(self)
    }

    /// Fetch prices for a token pair from all supported DEXes (Jupiter, Raydium, Orca)
    pub async fn fetch_all_dex_prices(
        &self,
        rpc_client: &RpcClient,
        input_token: &str,
        output_token: &str,
        raydium_pool: Option<&Pubkey>,
        orca_pool: Option<&Pubkey>,
    ) -> HashMap<String, Option<f64>> {
        let mut prices = HashMap::new();

        // Jupiter (API-based)
        if let Some(jupiter) = &self.jupiter_client {
            match jupiter.get_price(input_token, output_token).await {
                Ok(price) => { prices.insert("jupiter".to_string(), Some(price)); },
                Err(e) => { warn!("Jupiter price fetch failed: {}", e); prices.insert("jupiter".to_string(), None); }
            }
        }

        // Raydium (on-chain)
        if let Some(pool_addr) = raydium_pool {
            let input_pub = Pubkey::from_str(input_token).ok();
            let output_pub = Pubkey::from_str(output_token).ok();
            if let (Some(input_pub), Some(output_pub)) = (input_pub, output_pub) {
                match fetch_raydium_pool_price(rpc_client, pool_addr, &input_pub, &output_pub).await {
                    Ok(Some(price)) => { prices.insert("raydium".to_string(), Some(price)); },
                    Ok(None) => { prices.insert("raydium".to_string(), None); },
                    Err(e) => { warn!("Raydium price fetch failed: {}", e); prices.insert("raydium".to_string(), None); }
                }
            } else {
                prices.insert("raydium".to_string(), None);
            }
        }

        // Orca (on-chain)
        if let Some(pool_addr) = orca_pool {
            let input_pub = Pubkey::from_str(input_token).ok();
            let output_pub = Pubkey::from_str(output_token).ok();
            if let (Some(input_pub), Some(output_pub)) = (input_pub, output_pub) {
                match fetch_orca_whirlpool_price(rpc_client, pool_addr, &input_pub, &output_pub).await {
                    Ok(Some(price)) => { prices.insert("orca".to_string(), Some(price)); },
                    Ok(None) => { prices.insert("orca".to_string(), None); },
                    Err(e) => { warn!("Orca price fetch failed: {}", e); prices.insert("orca".to_string(), None); }
                }
            } else {
                prices.insert("orca".to_string(), None);
            }
        }

        prices
    }
}

/// Fetch the price of a Raydium pool given the pool address and token mints.
/// Returns the price as output_token per input_token (e.g., USDC per SOL).
/// This is a minimal implementation and assumes the pool follows Raydium's AMM layout.
pub async fn fetch_raydium_pool_price(
    rpc_client: &RpcClient,
    pool_address: &Pubkey,
    input_token: &Pubkey,
    output_token: &Pubkey,
) -> Result<Option<f64>> {
    // Fetch the pool account data
    let account = match rpc_client.get_account(pool_address) {
        Ok(acc) => acc,
        Err(e) => {
            warn!("Failed to fetch Raydium pool account: {}", e);
            return Ok(None);
        }
    };
    // Raydium AMM pool layout: coin_vault, pc_vault, etc.
    // For simplicity, we assume the first 64 bytes are coin_vault and pc_vault pubkeys,
    // and the next 16 bytes are coin_vault_amount and pc_vault_amount (u64 each).
    // In production, use the actual Raydium AMM layout struct.
    if account.data.len() < 160 {
        warn!("Raydium pool account data too short");
        return Ok(None);
    }
    // Offsets based on Raydium's AMMInfo struct (see Raydium SDK)
    let coin_vault_offset = 72;
    let pc_vault_offset = 104;
    let coin_vault_amount_offset = 136;
    let pc_vault_amount_offset = 144;
    let coin_vault = match account.data[coin_vault_offset..coin_vault_offset+32].try_into() {
        Ok(arr) => Pubkey::new_from_array(arr),
        Err(_) => {
            warn!("Failed to parse coin_vault pubkey from Raydium pool data");
            return Ok(None);
        }
    };
    let pc_vault = match account.data[pc_vault_offset..pc_vault_offset+32].try_into() {
        Ok(arr) => Pubkey::new_from_array(arr),
        Err(_) => {
            warn!("Failed to parse pc_vault pubkey from Raydium pool data");
            return Ok(None);
        }
    };
    let coin_vault_amount = u64::from_le_bytes(account.data[coin_vault_amount_offset..coin_vault_amount_offset+8].try_into().unwrap());
    let pc_vault_amount = u64::from_le_bytes(account.data[pc_vault_amount_offset..pc_vault_amount_offset+8].try_into().unwrap());
    // Determine which vault is input/output
    let (input_amount, output_amount) = if input_token == &coin_vault {
        (coin_vault_amount, pc_vault_amount)
    } else if input_token == &pc_vault {
        (pc_vault_amount, coin_vault_amount)
    } else {
        warn!("Input token does not match pool vaults");
        return Ok(None);
    };
    if input_amount == 0 {
        warn!("Input vault amount is zero");
        return Ok(None);
    }
    let price = output_amount as f64 / input_amount as f64;
    Ok(Some(price))
}

/// Fetch the price of an Orca Whirlpool pool given the pool address and token mints.
/// Returns the price as output_token per input_token (e.g., USDC per SOL).
/// This is a minimal implementation and assumes the pool follows Orca's Whirlpool layout.
pub async fn fetch_orca_whirlpool_price(
    rpc_client: &RpcClient,
    pool_address: &Pubkey,
    input_token: &Pubkey,
    output_token: &Pubkey,
) -> Result<Option<f64>> {
    // Fetch the pool account data
    let account = match rpc_client.get_account(pool_address) {
        Ok(acc) => acc,
        Err(e) => {
            warn!("Failed to fetch Orca Whirlpool pool account: {}", e);
            return Ok(None);
        }
    };
    // Orca Whirlpool layout: skip 8 bytes (anchor discriminator), then parse fields
    if account.data.len() < 300 {
        warn!("Orca Whirlpool account data too short");
        return Ok(None);
    }
    let data = &account.data[8..]; // skip anchor discriminator
    // Offsets based on Orca Whirlpool struct (see orca.rs and on-chain layout)
    let sqrt_price_offset = 48; // u128
    let token_mint_a_offset = 96; // Pubkey
    let token_mint_b_offset = 128; // Pubkey
    // Parse sqrt_price (u128, little endian)
    let sqrt_price = u128::from_le_bytes(data[sqrt_price_offset..sqrt_price_offset+16].try_into().unwrap());
    // Parse token mints
    let token_mint_a = match data[token_mint_a_offset..token_mint_a_offset+32].try_into() {
        Ok(arr) => Pubkey::new_from_array(arr),
        Err(_) => {
            warn!("Failed to parse token_mint_a pubkey from Orca Whirlpool data");
            return Ok(None);
        }
    };
    let token_mint_b = match data[token_mint_b_offset..token_mint_b_offset+32].try_into() {
        Ok(arr) => Pubkey::new_from_array(arr),
        Err(_) => {
            warn!("Failed to parse token_mint_b pubkey from Orca Whirlpool data");
            return Ok(None);
        }
    };
    // Compute price: price = (sqrt_price^2) / 2^128
    let price = (sqrt_price as f64).powi(2) / (2f64).powi(128);
    // Determine direction
    let result = if input_token == &token_mint_a && output_token == &token_mint_b {
        Some(price)
    } else if input_token == &token_mint_b && output_token == &token_mint_a {
        if price == 0.0 {
            None
        } else {
            Some(1.0 / price)
        }
    } else {
        warn!("Input/output tokens do not match pool mints");
        None
    };
    Ok(result)
}