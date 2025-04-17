/// Configuration for the ListenEngine
#[derive(Debug, Clone)]
pub struct ListenEngineConfig {
    pub rpc_url: String,
    pub commitment: String,
    pub ws_url: Option<String>,
}

impl Default for ListenEngineConfig {
    fn default() -> Self {
        Self {
            rpc_url: "https://api.mainnet-beta.solana.com".to_string(),
            commitment: "confirmed".to_string(),
            ws_url: None,
        }
    }
}