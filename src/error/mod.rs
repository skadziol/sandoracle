use std::fmt;
use reqwest::StatusCode;

mod utils;

pub type Result<T> = std::result::Result<T, SandoError>;

#[derive(Debug)]
pub enum SandoError {
    Config(config::ConfigError),
    Environment(String),
    SolanaRpc(String),
    Simulation(String),
    
    TransactionError {
        kind: TransactionErrorKind,
        message: String,
    },
    
    Strategy {
        kind: StrategyErrorKind,
        message: String,
    },
    
    Api {
        service: String,
        message: String,
        status: Option<u16>,
    },
    
    Telegram(String),
    
    RigAgent {
        kind: RigErrorKind,
        message: String,
    },
    
    Evaluation {
        kind: EvalErrorKind,
        message: String,
    },
    
    Io(std::io::Error),
    Serialization(serde_json::Error),
    HttpClient(reqwest::Error),
    Unknown(String),
    Timeout(String),
    
    HttpError {
        status: StatusCode,
        message: String,
    },
    
    NetworkError(String),
    Unexpected(String),
    ConfigError(String),
    DatabaseError(String),
    TransactionProcessingError(String),
    InternalError(String),
    EngineError(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TransactionErrorKind {
    InvalidSignature,
    InsufficientFunds,
    SimulationFailed,
    Timeout,
    RpcError,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StrategyErrorKind {
    InvalidParameters,
    InsufficientLiquidity,
    PriceImpactTooHigh,
    OpportunityExpired,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RigErrorKind {
    ApiError,
    InvalidResponse,
    ModelError,
    ContextTooLarge,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvalErrorKind {
    InvalidData,
    ProfitTooLow,
    RiskTooHigh,
    StrategyUnavailable,
    Other,
}

impl fmt::Display for SandoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SandoError::Config(err) => write!(f, "Configuration error: {}", err),
            SandoError::Environment(msg) => write!(f, "Environment error: {}", msg),
            SandoError::SolanaRpc(msg) => write!(f, "Solana RPC error: {}", msg),
            SandoError::TransactionError { kind, message } => {
                write!(f, "Transaction error: {} - {}", kind, message)
            }
            SandoError::Strategy { kind, message } => {
                write!(f, "MEV strategy error: {} - {}", kind, message)
            }
            SandoError::Api { service, message, status } => {
                if let Some(status) = status {
                    write!(f, "API error ({}): {} - {}", service, status, message)
                } else {
                    write!(f, "API error ({}): {}", service, message)
                }
            }
            SandoError::Telegram(msg) => write!(f, "Telegram notification error: {}", msg),
            SandoError::RigAgent { kind, message } => {
                write!(f, "RIG agent error: {} - {}", kind, message)
            }
            SandoError::Evaluation { kind, message } => {
                write!(f, "Opportunity evaluation error: {} - {}", kind, message)
            }
            SandoError::Io(err) => write!(f, "IO error: {}", err),
            SandoError::Serialization(err) => write!(f, "Serialization error: {}", err),
            SandoError::HttpClient(err) => write!(f, "HTTP client error: {}", err),
            SandoError::Unknown(msg) => write!(f, "Unknown error: {}", msg),
            SandoError::Timeout(msg) => write!(f, "Timeout error: {}", msg),
            SandoError::HttpError { status, message } => {
                write!(f, "HTTP error {}: {}", status, message)
            }
            SandoError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            SandoError::Unexpected(msg) => write!(f, "Unexpected error: {}", msg),
            SandoError::ConfigError(msg) => write!(f, "Invalid configuration: {}", msg),
            SandoError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            SandoError::TransactionProcessingError(msg) => write!(f, "Transaction processing error: {}", msg),
            SandoError::InternalError(msg) => write!(f, "Internal error: {}", msg),
            SandoError::EngineError(msg) => write!(f, "Engine error: {}", msg),
            SandoError::Simulation(msg) => write!(f, "Simulation error: {}", msg),
        }
    }
}

impl fmt::Display for TransactionErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSignature => write!(f, "Invalid signature"),
            Self::InsufficientFunds => write!(f, "Insufficient funds"),
            Self::SimulationFailed => write!(f, "Transaction simulation failed"),
            Self::Timeout => write!(f, "Transaction timeout"),
            Self::RpcError => write!(f, "RPC error"),
            Self::Other => write!(f, "Other transaction error"),
        }
    }
}

impl fmt::Display for StrategyErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameters => write!(f, "Invalid strategy parameters"),
            Self::InsufficientLiquidity => write!(f, "Insufficient liquidity"),
            Self::PriceImpactTooHigh => write!(f, "Price impact too high"),
            Self::OpportunityExpired => write!(f, "Opportunity expired"),
            Self::Other => write!(f, "Other strategy error"),
        }
    }
}

impl fmt::Display for RigErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiError => write!(f, "RIG API error"),
            Self::InvalidResponse => write!(f, "Invalid response from RIG"),
            Self::ModelError => write!(f, "Model error"),
            Self::ContextTooLarge => write!(f, "Context too large"),
            Self::Other => write!(f, "Other RIG error"),
        }
    }
}

impl fmt::Display for EvalErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData => write!(f, "Invalid evaluation data"),
            Self::ProfitTooLow => write!(f, "Profit below threshold"),
            Self::RiskTooHigh => write!(f, "Risk above threshold"),
            Self::StrategyUnavailable => write!(f, "Strategy unavailable"),
            Self::Other => write!(f, "Other evaluation error"),
        }
    }
}

// Helper functions for error conversion
impl SandoError {
    pub fn from_str(s: &str) -> Self {
        SandoError::Unknown(s.to_string())
    }

    pub fn from_string(s: String) -> Self {
        SandoError::Unknown(s)
    }

    pub fn transaction(kind: TransactionErrorKind, message: impl Into<String>) -> Self {
        SandoError::TransactionError {
            kind,
            message: message.into(),
        }
    }

    pub fn strategy(kind: StrategyErrorKind, message: impl Into<String>) -> Self {
        SandoError::Strategy {
            kind,
            message: message.into(),
        }
    }

    pub fn api(service: impl Into<String>, message: impl Into<String>, status: Option<u16>) -> Self {
        SandoError::Api {
            service: service.into(),
            message: message.into(),
            status,
        }
    }

    pub fn rig(kind: RigErrorKind, message: impl Into<String>) -> Self {
        SandoError::RigAgent {
            kind,
            message: message.into(),
        }
    }

    pub fn evaluation(kind: EvalErrorKind, message: impl Into<String>) -> Self {
        SandoError::Evaluation {
            kind,
            message: message.into(),
        }
    }

    pub fn is_retryable(&self) -> bool {
        match self {
            SandoError::TransactionError { 
                kind: TransactionErrorKind::Timeout | TransactionErrorKind::RpcError,
                ..
            } => true,
            SandoError::HttpError { status, .. } => *status >= StatusCode::from_u16(500).unwrap(),
            SandoError::RigAgent { 
                kind: RigErrorKind::ApiError,
                ..
            } => true,
            SandoError::NetworkError(_) => true,
            SandoError::Timeout(_) => true,
            _ => false,
        }
    }
}

// Implement conversion from string types
impl From<&str> for SandoError {
    fn from(s: &str) -> Self {
        SandoError::from_str(s)
    }
}

impl From<String> for SandoError {
    fn from(s: String) -> Self {
        SandoError::from_string(s)
    }
}

impl From<reqwest::Error> for SandoError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            SandoError::Timeout(format!("Request timed out: {}", err))
        } else if let Some(status) = err.status() {
            SandoError::HttpError {
                status,
                message: format!("{}", err),
            }
        } else {
            SandoError::HttpClient(err)
        }
    }
}

impl From<config::ConfigError> for SandoError {
    fn from(err: config::ConfigError) -> Self {
        SandoError::Config(err)
    }
}

impl From<std::io::Error> for SandoError {
    fn from(err: std::io::Error) -> Self {
        SandoError::Io(err)
    }
}

impl From<serde_json::Error> for SandoError {
    fn from(err: serde_json::Error) -> Self {
        SandoError::Serialization(err)
    }
}

// Add EngineError conversion for a generic error type
pub fn from_engine_error<E: std::fmt::Display>(err: E) -> SandoError {
    SandoError::EngineError(err.to_string())
}

pub fn should_retry(err: &SandoError) -> bool {
    match err {
        SandoError::TransactionError { kind, .. } => matches!(
            kind,
            TransactionErrorKind::Timeout | TransactionErrorKind::RpcError
        ),
        SandoError::HttpError { status, .. } => *status >= StatusCode::from_u16(500).unwrap(),
        SandoError::RigAgent { kind, .. } => matches!(kind, RigErrorKind::ApiError),
        SandoError::NetworkError(_) => true,
        SandoError::Timeout(_) => true,
        _ => false,
    }
}

pub fn is_retryable(err: &SandoError) -> bool {
    should_retry(err)
}

// Implement From trait for anyhow::Error
impl From<anyhow::Error> for SandoError {
    fn from(err: anyhow::Error) -> Self {
        SandoError::InternalError(format!("{:?}", err)) // Use debug format for more details
    }
}

// Standard error trait implementation
impl std::error::Error for SandoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SandoError::Config(err) => Some(err),
            SandoError::Environment(_) => None,
            SandoError::SolanaRpc(_) => None,
            SandoError::Simulation(_) => None,
            SandoError::TransactionError { .. } => None,
            SandoError::Strategy { .. } => None,
            SandoError::Api { .. } => None,
            SandoError::Telegram(_) => None,
            SandoError::RigAgent { .. } => None,
            SandoError::Evaluation { .. } => None,
            SandoError::Io(err) => Some(err),
            SandoError::Serialization(err) => Some(err),
            SandoError::HttpClient(err) => Some(err),
            SandoError::Unknown(_) => None,
            SandoError::Timeout(_) => None,
            SandoError::HttpError { .. } => None,
            SandoError::NetworkError(_) => None,
            SandoError::Unexpected(_) => None,
            SandoError::ConfigError(_) => None,
            SandoError::DatabaseError(_) => None,
            SandoError::TransactionProcessingError(_) => None,
            SandoError::InternalError(_) => None,
            SandoError::EngineError(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion() {
        let err: Result<()> = Err(SandoError::from("test error"));
        assert!(matches!(err, Err(SandoError::Unknown(_))));

        let err: Result<()> = Err("test error".into());
        assert!(matches!(err, Err(SandoError::Unknown(_))));

        let err = SandoError::strategy(
            StrategyErrorKind::InsufficientLiquidity,
            "Not enough liquidity in pool",
        );
        assert!(matches!(err, SandoError::Strategy { .. }));
    }

    #[test]
    fn test_retryable_errors() {
        let err = SandoError::transaction(TransactionErrorKind::Timeout, "Transaction timed out");
        assert!(err.is_retryable());

        let err = SandoError::api("Orca", "Internal server error", Some(500));
        assert!(err.is_retryable());

        let err = SandoError::transaction(TransactionErrorKind::InvalidSignature, "Invalid signature");
        assert!(!err.is_retryable());
    }
} 