use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, Instant};
use tokio::sync::RwLock;
use crate::error::{Result, SandoError};
use crate::jupiter_client::{Jupiter, TokenInfo};
use tracing::{info, error, warn, debug};
use futures::future;
use tokio::time::sleep;
use std::cmp::min;
use solana_client::rpc_client::RpcClient;
use solana_sdk::pubkey::Pubkey;
use std::str::FromStr;

/// Represents the relevant state of a Raydium AMM V4 pool.
#[derive(Debug, Clone, Copy)]
pub struct RaydiumAmmState {
    pub coin_vault_amount: u64,
    pub pc_vault_amount: u64,
    pub coin_mint: Pubkey, // Mint of the coin vault
    pub pc_mint: Pubkey,   // Mint of the pc vault
    // Add other relevant fields like lp_mint, fees later if needed
}

/// Represents the relevant state of an Orca Whirlpool.
#[derive(Debug, Clone, Copy)]
pub struct WhirlpoolState {
    pub sqrt_price: u128,
    pub liquidity: u128,
    pub tick_current_index: i32,
    pub token_mint_a: Pubkey,
    pub token_mint_b: Pubkey,
    // Add other fields like fee_rate, protocol_fee_rate if needed later
}

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
    
    /// RpcClient for on-chain fetching
    rpc_client: Arc<RpcClient>,
}

impl MarketDataCollector {
    /// Create a new market data collector
    pub fn new(
        price_cache: Arc<RwLock<HashMap<String, f64>>>,
        historical_data: Arc<RwLock<HashMap<String, Vec<f64>>>>,
        rpc_endpoint: &str,
    ) -> Result<Self> {
        // Convert existing price cache to new format with TTL
        let new_cache = Arc::new(RwLock::new(HashMap::new()));
        let market_data_cache = Arc::new(RwLock::new(HashMap::new()));
        let last_batch_update = Arc::new(RwLock::new(None));
        let supported_tokens = Arc::new(RwLock::new(HashMap::new()));
        
        // Initialize RpcClient
        let rpc_client = Arc::new(RpcClient::new(rpc_endpoint.to_string()));
        
        let collector = Self {
            price_cache: new_cache,
            historical_data,
            jupiter_client: None,
            default_ttl: 300, // 5 minutes default TTL
            market_data_cache,
            last_batch_update,
            max_retries: 3,
            supported_tokens,
            rpc_client,
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
        
        Ok(collector)
    }

    /// Initialize with Jupiter API for real-time price data
    pub fn with_jupiter(mut self, jupiter_api_url: &str) -> Result<Self> {
        // Initialize Jupiter client
        self.jupiter_client = Some(Jupiter::new(jupiter_api_url));
        
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
    
    /// Fetch TokenInfo for a given mint address, using cache.
    /// NOTE: Need a dedicated cache for TokenInfo.
    async fn get_token_info(&self, token_mint: &str) -> Result<Option<TokenInfo>> {
        // TODO: Implement caching for TokenInfo similar to price_cache
        // For now, fetch directly every time (inefficient)
        warn!("TokenInfo fetching is not cached!");
        let token_list = Jupiter::get_token_list().await?;
        
        // Call the now public find_token_info
        match Jupiter::find_token_info(&token_list, token_mint) {
            Ok(info) => Ok(Some(info.clone())), // Found: wrap in Ok(Some(...))
            Err(e) => {
                // Check if the error is specifically "token not found"
                if e.to_string().contains("not found in Jupiter list") {
                    Ok(None) // Not found: return Ok(None)
                } else {
                     Err(e.into()) // Other error type: convert to SandoError and propagate
                }
            }
        }
    }

    /// Discover potential pools and fetch token info.
    async fn discover_pools_and_tokens(
        &self,
        input_mint_str: &str,
        output_mint_str: &str,
    ) -> Result<(
        Option<Pubkey>, // Raydium Pool
        Option<Pubkey>, // Orca Pool
        Option<TokenInfo>, // Input Token Info
        Option<TokenInfo>, // Output Token Info
    )> {
        let jupiter_client = self.get_jupiter_client().ok_or_else(|| 
            SandoError::ConfigError("Jupiter client not initialized".to_string())
        )?;

        let nominal_amount = 1_000_000_000; 

        // Fetch token info first (could be parallelized with quote)
        let input_token_info = self.get_token_info(input_mint_str).await?;
        let output_token_info = self.get_token_info(output_mint_str).await?;

        // Fetch quote for pool discovery
        match Jupiter::fetch_quote(input_mint_str, output_mint_str, nominal_amount).await {
            Ok(quote_response) => {
                let mut raydium_pool: Option<Pubkey> = None;
                let mut orca_pool: Option<Pubkey> = None;

                for route in quote_response.route_plan {
                    if raydium_pool.is_some() && orca_pool.is_some() {
                        break; // Found both
                    }

                    if let Some(label) = route.swap_info.label {
                        // Attempt to parse the amm_key
                        let amm_key_pk = match Pubkey::from_str(&route.swap_info.amm_key) {
                            Ok(pk) => pk,
                            Err(_) => {
                                warn!(amm_key=%route.swap_info.amm_key, "Failed to parse AMM key from Jupiter route");
                                continue;
                            }
                        };

                        // Check labels (case-insensitive)
                        if raydium_pool.is_none() && label.to_lowercase().contains("raydium") {
                           debug!(input=%input_mint_str, output=%output_mint_str, pool=%amm_key_pk, "Discovered potential Raydium pool via Jupiter");
                            raydium_pool = Some(amm_key_pk);
                        } else if orca_pool.is_none() && label.to_lowercase().contains("orca") {
                           debug!(input=%input_mint_str, output=%output_mint_str, pool=%amm_key_pk, "Discovered potential Orca pool via Jupiter");
                            orca_pool = Some(amm_key_pk);
                        }
                        // TODO: Add checks for specific Raydium AMM V4 / Orca Whirlpool programs if needed
                    }
                }
                Ok((raydium_pool, orca_pool, input_token_info, output_token_info))
            }
            Err(e) => {
                warn!(input=%input_mint_str, output=%output_mint_str, error=%e, "Failed quote for pool discovery");
                // Still return token info even if pool discovery failed
                Ok((None, None, input_token_info, output_token_info)) 
            }
        }
    }

    /// Get full market data for a token pair, incorporating on-chain pool state.
    pub async fn get_market_data(
        &self, 
        input_token_mint: &str, 
        output_token_mint: &str,
    ) -> Result<Option<MarketData>> {
        let cache_key = format!("{}-{}", input_token_mint, output_token_mint);

        // Check full data cache first
        {
            let cache = self.market_data_cache.read().await;
            if let Some(entry) = cache.get(&cache_key) {
                if !entry.is_expired() {
                    debug!(key=%cache_key, "Using cached MarketData");
                    return Ok(Some(entry.data.clone()));
                }
            }
        }
        
        // Discover pools and fetch token info
        let (discovered_raydium_pool, discovered_orca_pool, input_token_info, output_token_info) = 
            self.discover_pools_and_tokens(input_token_mint, output_token_mint)
                .await
                .unwrap_or((None, None, None, None)); // Handle potential error

        // Check if we have essential info (decimals)
        let Some(input_info) = input_token_info else {
            warn!(token=%input_token_mint, "Could not find token info (decimals) for input token");
            return Ok(None);
        };
        let Some(output_info) = output_token_info else {
            warn!(token=%output_token_mint, "Could not find token info (decimals) for output token");
            return Ok(None);
        };

        // Fetch price (primarily rely on Jupiter for now)
        let jupiter_price_opt = self
            .fetch_real_market_data(input_token_mint, output_token_mint)
            .await?;

        // If Jupiter price is unavailable, we cannot proceed reliably yet
        let price = if let Some(p) = jupiter_price_opt {
            p
        } else {
            debug!(input=%input_token_mint, output=%output_token_mint, "No price found via Jupiter, cannot build MarketData");
            return Ok(None); 
        };

        // Now 'price' is available in the rest of the function scope

        let mut source = "jupiter".to_string();
        let mut confidence = 0.90; // Base confidence from Jupiter
        let mut on_chain_liquidity: Option<f64> = None;

        // Define stablecoin mint for USD price fetching
        const USDC_MINT: &str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";

        // Fetch USD price for input token (needed for TVL calculation)
        let input_token_usd_price_opt = self.fetch_real_market_data(input_token_mint, USDC_MINT).await?;

        // --- Try fetching Orca Whirlpool State --- 
        if let Some(pool_addr) = discovered_orca_pool.as_ref() { 
            match fetch_orca_whirlpool_state(&self.rpc_client, pool_addr).await {
                Ok(Some(state)) => {
                    debug!(pool=%pool_addr, "Fetched Orca state: Liquidity={}, Tick={}", state.liquidity, state.tick_current_index);
                    
                    // Attempt rough TVL calculation for Orca (less accurate for CLMM)
                    if let Some(input_usd_price) = input_token_usd_price_opt {
                         // Assuming token_a is input, token_b is output - Needs verification!
                         // Requires fetch_orca_whirlpool_state to return verified mints
                        let token_a_amount = state.liquidity as f64; // THIS IS WRONG FOR CLMM TVL
                        let token_b_amount = token_a_amount * price; // Also likely wrong
                        let token_a_value = (token_a_amount / 10f64.powi(input_info.decimals)) * input_usd_price;
                        let token_b_value = (token_b_amount / 10f64.powi(output_info.decimals)) * (input_usd_price / price); // Estimate output price
                        on_chain_liquidity = Some(token_a_value + token_b_value);
                        warn!(pool=%pool_addr, "Calculated rough TVL for Orca pool (CLMM), may be inaccurate.");
                    } else {
                        // Fallback placeholder if USD price fails
                        on_chain_liquidity = Some(-1.0); // Indicate found but not calculated
                         warn!(pool=%pool_addr, "Could not fetch input token USD price, using placeholder liquidity for Orca pool.");
                    }
                    
                    source = format!("{}+orca_state", source);
                    confidence = (confidence + 0.95) / 2.0; // Slightly lower confidence for Orca TVL estimate
                }
                Ok(None) => {
                    debug!(pool=%pool_addr, "No Orca state data found or account invalid.");
                }
                Err(e) => {
                    warn!(pool=%pool_addr, error=%e, "Error fetching Orca state");
                }
            }
        }

        // --- Try fetching Raydium AMM State --- 
        // Only fetch if Orca didn't provide a valid liquidity estimate
        if on_chain_liquidity.is_none() || on_chain_liquidity == Some(-1.0) {
             if let Some(pool_addr) = discovered_raydium_pool.as_ref() { 
                 // Need mint Pubkeys for price direction check in fetch_raydium_pool_price (even though flawed)
                 let _input_pk = Pubkey::from_str(input_token_mint).ok(); // Mark as unused for now
                 let output_pk = Pubkey::from_str(output_token_mint).ok();

                 // Fetch state first
                 match fetch_raydium_amm_state(&self.rpc_client, pool_addr).await {
                     Ok(Some(state)) => {
                         debug!(pool=%pool_addr, "Fetched Raydium state: Coin={}, PC={}", state.coin_vault_amount, state.pc_vault_amount);
                         
                         // Calculate Raydium TVL
                         if let Some(input_usd_price) = input_token_usd_price_opt {
                            // Assume coin_mint is input_token for calculation - NEEDS PARSING
                            let coin_decimals = input_info.decimals;
                            let pc_decimals = output_info.decimals;

                            let coin_value_usd = (state.coin_vault_amount as f64 / 10f64.powi(coin_decimals)) * input_usd_price;
                            // Estimate pc_price_usd = input_usd_price / pair_price (output_per_input)
                            let pc_price_usd = input_usd_price / price;
                            let pc_value_usd = (state.pc_vault_amount as f64 / 10f64.powi(pc_decimals)) * pc_price_usd;
                            
                            on_chain_liquidity = Some(coin_value_usd + pc_value_usd);
                            debug!(pool=%pool_addr, "Calculated Raydium TVL: {:.2} USD", on_chain_liquidity.unwrap_or(0.0));
                         } else {
                            on_chain_liquidity = Some(-1.0); // Indicate found but not calculated
                            warn!(pool=%pool_addr, "Could not fetch input token USD price, using placeholder liquidity for Raydium pool.");
                         }
                         
                         source = format!("{}+raydium_state", source);
                         confidence = (confidence + 0.98) / 2.0; // Higher confidence for Raydium TVL estimate
                     }
                     Ok(None) => {
                         debug!(pool=%pool_addr, "No Raydium state data found or account invalid.");
                     }
                     Err(e) => {
                         warn!(pool=%pool_addr, error=%e, "Error fetching Raydium state");
                     }
                 }
            }
        }

        // Build MarketData
        let history = self.get_token_history(input_token_mint).await.unwrap_or_default();
            
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            // Calculate price change if we have history
            let price_change = if history.len() >= 2 {
            let oldest = history.first().unwrap_or(&price); // 'price' is accessible here
            ((price - oldest) / oldest.max(1e-9)) * 100.0 // 'price' is accessible here
            } else {
                0.0
            };
            
            // Calculate volatility if we have enough history
            let volatility = if history.len() >= 3 {
                history.windows(2)
                .map(|w| ((w[1] - w[0]) / w[0].max(1e-9)).abs()) 
                    .sum::<f64>() / (history.len() - 1) as f64
            } else {
                0.01 // Default low volatility
            };
            
            let market_data = MarketData {
            price, // 'price' is accessible here
            market_cap: 0.0,      
            volume_24h: 0.0,      
                price_change_24h: price_change,
            liquidity: on_chain_liquidity.unwrap_or(0.0), 
                volatility,
                last_update: now,
                expires_at: Some(SystemTime::now() + Duration::from_secs(self.default_ttl)),
            source, 
            confidence, 
            };
            
            // Update the cache
            {
                let mut cache = self.market_data_cache.write().await;
            cache.insert(cache_key, CacheEntry::new(market_data.clone(), self.default_ttl));
            }
            
        Ok(Some(market_data))
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
                match fetch_raydium_pool_price(&self.rpc_client, pool_addr, &input_pub, &output_pub).await {
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
                match fetch_orca_whirlpool_price(&self.rpc_client, pool_addr, &input_pub, &output_pub).await {
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

/// Fetch the state of a Raydium AMM V4 pool given the pool address.
/// Returns vault amounts and mints.
/// Assumes the account layout follows Raydium's standard AMM V4 structure.
pub async fn fetch_raydium_amm_state(
    rpc_client: &RpcClient,
    pool_address: &Pubkey,
) -> Result<Option<RaydiumAmmState>> {
    let account = match rpc_client.get_account(pool_address) {
        Ok(acc) => acc,
        Err(e) => {
            warn!("Failed to fetch Raydium pool account {}: {}", pool_address, e);
            return Ok(None);
        }
    };

    // Based on common Raydium AMM V4 layout - VERIFY OFFSETS
    // https://github.com/raydium-io/raydium-sdk/blob/master/src/constants/layouts.ts
    const DATA_START: usize = 52; // Skip padding/metadata
    const MIN_DATA_LEN: usize = 400; // Rough estimate based on offsets

    if account.data.len() < MIN_DATA_LEN {
        warn!(
            "Raydium pool account data for {} too short: {} bytes",
            pool_address,
            account.data.len()
        );
        return Ok(None);
    }
    
    let data = &account.data; // Use full data slice

    // Precise offsets based on typical AMM V4 layout structure
    let coin_mint_offset = 75; // pool_coin_token_account -> mint @ 32 + 8 + 32 + 3 = 75?
    let pc_mint_offset = 107;  // pool_pc_token_account -> mint @ 75 + 32 = 107?
    let coin_vault_amount_offset = 136; // amount in pool_coin_token_account?
    let pc_vault_amount_offset = 168; // amount in pool_pc_token_account?
    // NOTE: These offsets are highly speculative without proper struct def or SDK.
    // The stub function used different offsets (e.g., 136/144 for amounts). Let's stick to those for now.
    let coin_vault_amount_offset_stub = 136;
    let pc_vault_amount_offset_stub = 144;
    // Let's try getting mints from standard locations if possible
    let coin_mint_from_vault_offset = 72 + 32; // coin_vault_mint offset in AMMInfo? No standard here.
    // Reverting to simpler extraction based *only* on stub offsets for amounts
    // Need proper SDK/layout parsing for mints.

    // Helper to parse bytes safely
    let parse_bytes = |offset: usize, len: usize, field_name: &str| -> Result<&[u8]> {
        data.get(offset..offset + len)
            .ok_or_else(|| SandoError::DataProcessing(format!(
                "Failed to slice bytes for {} at offset {} (len {}) in Raydium pool {}",
                field_name, offset, len, pool_address
            )))
    };

    // Parse vault amounts using offsets from the original stub function
    let coin_vault_amount = u64::from_le_bytes(
        parse_bytes(coin_vault_amount_offset_stub, 8, "coin_vault_amount")?.try_into().unwrap()
    );
    let pc_vault_amount = u64::from_le_bytes(
        parse_bytes(pc_vault_amount_offset_stub, 8, "pc_vault_amount")?.try_into().unwrap()
    );

    // *** CANNOT RELIABLY GET MINTS WITHOUT PROPER PARSING/SDK ***
    // For now, return state without mints, requiring caller to know them.
    // Ideally, parse coin_mint_pk and pc_mint_pk from the state.

    Ok(Some(RaydiumAmmState {
        coin_vault_amount,
        pc_vault_amount,
        coin_mint: Pubkey::default(), // Placeholder - Needs real parsing
        pc_mint: Pubkey::default(),   // Placeholder - Needs real parsing
    }))
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
    match fetch_raydium_amm_state(rpc_client, pool_address).await? {
        Some(state) => {
             // Determine which vault is input/output based on the *caller provided* mints
             // This is fragile without parsing mints from the state itself.
             // Assuming caller knows coin_mint maps to coin_vault_amount etc.
            let (input_amount, output_amount) = 
                if input_token == &state.coin_mint { // Requires state.coin_mint to be parsed correctly
                     warn!("Raydium state parsing incomplete: Assuming input {} is coin_mint", input_token);
                    (state.coin_vault_amount, state.pc_vault_amount)
                } else if input_token == &state.pc_mint { // Requires state.pc_mint to be parsed correctly
                     warn!("Raydium state parsing incomplete: Assuming input {} is pc_mint", input_token);
                    (state.pc_vault_amount, state.coin_vault_amount)
    } else {
                    // If mints weren't parsed from state (current situation), we cannot reliably determine direction.
                    // Fallback: Try matching based on amounts? Very unreliable.
                    warn!(
                        "Cannot determine Raydium price direction for pool {}. Input token {} not matched to parsed mints (currently placeholders).",
                        pool_address, input_token
                    );
        return Ok(None);
    };

    if input_amount == 0 {
                warn!("Input vault amount is zero for pool {}", pool_address);
        return Ok(None);
    }
    let price = output_amount as f64 / input_amount as f64;
    Ok(Some(price))
        }
        None => Ok(None),
    }
}

/// Fetch the state of an Orca Whirlpool pool given the pool address.
/// Returns relevant fields like sqrt_price, liquidity, current tick.
/// Assumes the account layout follows Orca's Whirlpool standard structure post-Anchor discriminator.
pub async fn fetch_orca_whirlpool_state(
    rpc_client: &RpcClient,
    pool_address: &Pubkey,
) -> Result<Option<WhirlpoolState>> {
    // Fetch the pool account data
    let account = match rpc_client.get_account(pool_address) {
        Ok(acc) => acc,
        Err(e) => {
            warn!("Failed to fetch Orca Whirlpool pool account {}: {}", pool_address, e);
            // Return Ok(None) instead of error to indicate data not available
            return Ok(None);
        }
    };

    // Standard Anchor discriminator length
    const ANCHOR_DISCRIMINATOR_LEN: usize = 8;

    // Check if data is long enough for the fields we need + discriminator
    // Need up to token_mint_b offset + size = 128 + 32 = 160
    if account.data.len() < ANCHOR_DISCRIMINATOR_LEN + 160 { 
        warn!(
            "Orca Whirlpool account data for {} too short: {} bytes",
            pool_address,
            account.data.len()
        );
        return Ok(None);
    }

    let data = &account.data[ANCHOR_DISCRIMINATOR_LEN..]; // skip anchor discriminator

    // Offsets relative to data start (after discriminator)
    // Based on common Whirlpool struct layouts - VERIFY THESE if possible
    let sqrt_price_offset = 48; // u128 @ 48 + 8 = 56
    let liquidity_offset = 64;  // u128 @ 64 + 8 = 72
    let tick_current_index_offset = 80; // i32 @ 80 + 8 = 88
    let token_mint_a_offset = 96; // Pubkey @ 96 + 8 = 104
    let token_mint_b_offset = 128; // Pubkey @ 128 + 8 = 136

    // Helper to parse bytes safely
    let parse_bytes = |offset: usize, len: usize, field_name: &str| -> Result<&[u8]> {
        data.get(offset..offset + len)
            .ok_or_else(|| SandoError::DataProcessing(format!(
                "Failed to slice bytes for {} at offset {} (len {}) in pool {}",
                field_name, offset, len, pool_address
            )))
    };

    // Helper to parse Pubkey safely
    let parse_pubkey = |offset: usize, field_name: &str| -> Result<Pubkey> {
        let bytes = parse_bytes(offset, 32, field_name)?;
        bytes.try_into()
            .map(Pubkey::new_from_array)
            .map_err(|_| SandoError::DataProcessing(format!(
                "Failed to convert bytes to Pubkey for {} in pool {}",
                field_name, pool_address
            )))
    };

    // Parse fields using safe helpers
    let sqrt_price = u128::from_le_bytes(
        parse_bytes(sqrt_price_offset, 16, "sqrt_price")?.try_into().unwrap() // Should not fail if slice is correct length
    );
    let liquidity = u128::from_le_bytes(
        parse_bytes(liquidity_offset, 16, "liquidity")?.try_into().unwrap()
    );
    let tick_current_index = i32::from_le_bytes(
        parse_bytes(tick_current_index_offset, 4, "tick_current_index")?.try_into().unwrap()
    );
    let token_mint_a = parse_pubkey(token_mint_a_offset, "token_mint_a")?;
    let token_mint_b = parse_pubkey(token_mint_b_offset, "token_mint_b")?;

    Ok(Some(WhirlpoolState {
        sqrt_price,
        liquidity,
        tick_current_index,
        token_mint_a,
        token_mint_b,
    }))
}

// Keep the old price function for now, maybe mark as deprecated or remove later
/// Fetch the price of an Orca Whirlpool pool given the pool address and token mints.
/// Returns the price as output_token per input_token (e.g., USDC per SOL).
/// This is a minimal implementation and assumes the pool follows Orca's Whirlpool layout.
pub async fn fetch_orca_whirlpool_price(
    rpc_client: &RpcClient,
    pool_address: &Pubkey,
    input_token: &Pubkey,
    output_token: &Pubkey,
) -> Result<Option<f64>> {
    match fetch_orca_whirlpool_state(rpc_client, pool_address).await? {
        Some(state) => {
    // Compute price: price = (sqrt_price^2) / 2^128
            let price = (state.sqrt_price as f64).powi(2) / (2f64).powi(128);
            
    // Determine direction
            if input_token == &state.token_mint_a && output_token == &state.token_mint_b {
                Ok(Some(price))
            } else if input_token == &state.token_mint_b && output_token == &state.token_mint_a {
        if price == 0.0 {
                    Ok(None)
        } else {
                    Ok(Some(1.0 / price))
        }
    } else {
                warn!("Input/output tokens {}/{} do not match pool mints {}/{} for {}", 
                    input_token, output_token, state.token_mint_a, state.token_mint_b, pool_address);
                Ok(None)
            }
        }
        None => Ok(None),
    }
}