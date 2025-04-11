use crate::engine::error::EngineError;
use crate::redis::client::{make_redis_client, RedisClient};
use crate::redis::subscriber::{make_redis_subscriber, PriceUpdate, RedisSubscriber};
use crate::market_data::{MarketDataCollector, MarketData};

pub struct Engine {
    pub redis: Arc<RedisClient>,
    pub redis_sub: Arc<RedisSubscriber>,
    pub privy: Arc<Privy>,

    // Current market state
    price_cache: Arc<RwLock<HashMap<String, f64>>>,
    market_data_collector: Arc<MarketDataCollector>,
    processing_pipelines: Arc<Mutex<HashSet<String>>>,
    active_pipelines: Arc<DashMap<String, HashSet<String>>>, // asset -> pipeline ids
    shutdown_signal: Arc<Notify>,                            // Used to signal shutdown
    pending_tasks: Arc<AtomicUsize>, // Track number of running pipeline evaluations
}

impl Clone for Engine {
    fn clone(&self) -> Self {
        Self {
            redis: self.redis.clone(),
            redis_sub: self.redis_sub.clone(),
            privy: self.privy.clone(),
            price_cache: self.price_cache.clone(),
            market_data_collector: self.market_data_collector.clone(),
            processing_pipelines: self.processing_pipelines.clone(),
            active_pipelines: self.active_pipelines.clone(),
            shutdown_signal: self.shutdown_signal.clone(),
            pending_tasks: self.pending_tasks.clone(),
        }
    }
}

impl Engine {
    pub async fn new(config: PrivyConfig) -> Result<Self> {
        let redis = Arc::new(make_redis_client(&config.redis_url).await?);
        let redis_sub = Arc::new(make_redis_subscriber(&config.redis_url).await?);
        let privy = Arc::new(Privy::new(config));
        
        let price_cache = Arc::new(RwLock::new(HashMap::new()));
        let historical_data = Arc::new(RwLock::new(HashMap::new()));
        
        let market_data_collector = Arc::new(MarketDataCollector::new(
            price_cache.clone(),
            historical_data,
        ));

        Ok(Self {
            redis,
            redis_sub,
            privy,
            price_cache,
            market_data_collector,
            processing_pipelines: Arc::new(Mutex::new(HashSet::new())),
            active_pipelines: Arc::new(DashMap::new()),
            shutdown_signal: Arc::new(Notify::new()),
            pending_tasks: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Get market data for a specific asset
    pub async fn get_market_data(&self, asset: &str) -> Result<Option<MarketData>> {
        self.market_data_collector.get_market_data(asset).await
    }

    /// Get market data for all assets
    pub async fn get_all_market_data(&self) -> Result<HashMap<String, MarketData>> {
        self.market_data_collector.get_all_market_data().await
    }
} 