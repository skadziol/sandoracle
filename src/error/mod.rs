use thiserror::Error;
use std::fmt;
use reqwest::StatusCode;

mod utils;
pub use utils::*;

#[derive(Error, Debug)]
pub enum SandoError {
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("Environment error: {0}")]
    Environment(String),

    #[error("Solana RPC error: {0}")]
    SolanaRpc(String),

    #[error("Transaction error: {kind} - {message}")]
    TransactionError {
        kind: TransactionErrorKind,
        message: String,
    },

    #[error("MEV strategy error: {kind} - {message}")]
    Strategy {
        kind: StrategyErrorKind,
        message: String,
    },

    #[error("API error: {service} - {message}")]
    Api {
        service: String,
        message: String,
        status: Option<u16>,
    },

    #[error("Telegram notification error: {0}")]
    Telegram(String),

    #[error("RIG agent error: {kind} - {message}")]
    RigAgent {
        kind: RigErrorKind,
        message: String,
    },

    #[error("Opportunity evaluation error: {kind} - {message}")]
    Evaluation {
        kind: EvalErrorKind,
        message: String,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP client error: {0}")]
    HttpClient(#[from] reqwest::Error),

    #[error("Unknown error: {0}")]
    Unknown(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("HTTP error: {status} - {message}")]
    HttpError {
        status: StatusCode,
        message: String,
    },

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Unexpected error: {0}")]
    Unexpected(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Transaction processing error: {0}")]
    TransactionProcessingError(String),

    #[error("Internal error: {0}")]
    InternalError(String),
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

pub type Result<T> = std::result::Result<T, SandoError>;

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
        matches!(
            self,
            SandoError::TransactionError {
                kind: TransactionErrorKind::Timeout | TransactionErrorKind::RpcError,
                ..
            } | SandoError::Api { status: Some(status), .. } if status >= &500
            | SandoError::RigAgent {
                kind: RigErrorKind::ApiError,
                ..
            }
        )
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
        if let Some(status) = err.status() {
            SandoError::HttpError {
                status,
                message: err.to_string(),
            }
        } else {
            SandoError::NetworkError(err.to_string())
        }
    }
}

pub fn should_retry(err: &SandoError) -> bool {
    match err {
        SandoError::TransactionError { kind, .. } => matches!(
            kind,
            TransactionErrorKind::Timeout | TransactionErrorKind::RpcError
        ),
        SandoError::HttpError { status, .. } => *status >= StatusCode::from_u16(500).unwrap(),
        _ => false,
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