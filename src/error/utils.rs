use super::{Result, SandoError, TransactionErrorKind, StrategyErrorKind, RigErrorKind, EvalErrorKind};
use tokio::time::{sleep, Duration};
use tracing::{error, warn};

/// Retries a fallible async operation with exponential backoff.
///
/// # Arguments
/// * `operation` - An async closure that returns a Result
/// * `max_retries` - Maximum number of retry attempts
/// * `initial_delay_ms` - Initial delay in milliseconds, which doubles after each attempt
///
/// # Example
/// ```
/// let result = retry_with_backoff(
///     || async { 
///         // Your async operation here
///         Ok(()) 
///     },
///     3,
///     100
/// ).await;
/// ```
pub async fn retry_with_backoff<F, Fut, T, E>(
    operation: F,
    max_retries: u32,
    initial_delay_ms: u64,
) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut current_retry = 0;
    let mut delay_ms = initial_delay_ms;

    loop {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(e) => {
                if current_retry >= max_retries {
                    return Err(e);
                }
                warn!("Operation failed, retrying in {}ms. Error: {:?}", delay_ms, e);
                sleep(Duration::from_millis(delay_ms)).await;
                current_retry += 1;
                delay_ms *= 2;
            }
        }
    }
}

/// Logs an error with appropriate severity based on the error type.
/// 
/// # Arguments
/// * `error` - The SandoError to log
/// * `context` - Additional context about where/how the error occurred
pub fn log_error(error: &SandoError, context: &str) {
    match error {
        SandoError::TransactionError { kind, message } => {
            match kind {
                TransactionErrorKind::Timeout | TransactionErrorKind::RpcError => {
                    warn!("{} - Transaction error: {} - {}", context, kind, message);
                }
                _ => error!("{} - Transaction error: {} - {}", context, kind, message),
            }
        }
        SandoError::Strategy { kind, message } => {
            match kind {
                StrategyErrorKind::OpportunityExpired => {
                    warn!("{} - Strategy error: {} - {}", context, kind, message);
                }
                _ => error!("{} - Strategy error: {} - {}", context, kind, message),
            }
        }
        SandoError::RigAgent { kind, message } => {
            match kind {
                RigErrorKind::ApiError | RigErrorKind::ContextTooLarge => {
                    warn!("{} - RIG error: {} - {}", context, kind, message);
                }
                _ => error!("{} - RIG error: {} - {}", context, kind, message),
            }
        }
        SandoError::Evaluation { kind, message } => {
            match kind {
                EvalErrorKind::ProfitTooLow | EvalErrorKind::RiskTooHigh => {
                    warn!("{} - Evaluation error: {} - {}", context, kind, message);
                }
                _ => error!("{} - Evaluation error: {} - {}", context, kind, message),
            }
        }
        _ => error!("{} - {}", context, error),
    }
}

/// Converts a reqwest error to a SandoError with additional context.
/// 
/// # Arguments
/// * `error` - The reqwest error to convert
/// * `context` - Additional context about the request that failed
pub fn handle_reqwest_error(error: reqwest::Error, context: &str) -> SandoError {
    if error.is_timeout() {
        SandoError::Timeout(format!("{}: {}", context, error))
    } else if let Some(status) = error.status() {
        SandoError::HttpError {
            status,
            message: format!("{}: {}", context, error),
        }
    } else {
        SandoError::NetworkError(format!("{}: {}", context, error))
    }
}

/// Handles unexpected errors that should not occur in production.
/// Logs the error and returns a SandoError::Unknown.
/// 
/// # Arguments
/// * `error` - The unexpected error
/// * `context` - Additional context about where the error occurred
pub fn handle_unexpected_error<E: std::fmt::Display>(error: E, context: &str) -> SandoError {
    let message = format!("{}: {}", context, error);
    error!("Unexpected error: {}", message);
    SandoError::Unknown(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_with_backoff() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result: Result<(), String> = retry_with_backoff(
            move || {
                let counter = counter_clone.clone();
                async move {
                    let attempts = counter.fetch_add(1, Ordering::SeqCst);
                    if attempts < 2 {
                        Err("Not ready yet".to_string())
                    } else {
                        Ok(())
                    }
                }
            },
            3,
            10,
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_handle_reqwest_error() {
        let error = reqwest::Error::from(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "Request timed out",
        ));
        let result = handle_reqwest_error(error, "Failed to fetch data");
        match result {
            SandoError::Timeout(_) => (),
            _ => panic!("Expected Timeout error"),
        }
    }
} 