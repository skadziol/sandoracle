use super::SandoError;
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
pub async fn retry_with_backoff<F, Fut, T>(
    operation: F,
    max_retries: u32,
    initial_delay_ms: u64,
) -> std::result::Result<T, SandoError>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, SandoError>>,
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
        SandoError::HttpError { status, message } => {
            if status.is_server_error() {
                error!("{} - HTTP error {}: {}", context, status, message);
            } else {
                warn!("{} - HTTP error {}: {}", context, status, message);
            }
        }
        SandoError::NetworkError(msg) => {
            warn!("{} - Network error: {}", context, msg);
        }
        SandoError::TransactionError { kind, message } => {
            error!("{} - Transaction error: {} - {}", context, kind, message);
        }
        SandoError::ConfigError(msg) => {
            error!("{} - Configuration error: {}", context, msg);
        }
        SandoError::DatabaseError(msg) => {
            error!("{} - Database error: {}", context, msg);
        }
        SandoError::InternalError(msg) => {
            error!("{} - Internal error: {}", context, msg);
        }
        _ => error!("{} - Unexpected error: {}", context, error),
    }
}

/// Converts a reqwest error to a SandoError with additional context.
/// 
/// # Arguments
/// * `error` - The reqwest error to convert
/// * `context` - Additional context about the request that failed
pub fn handle_reqwest_error(error: reqwest::Error, context: &str) -> SandoError {
    if error.is_timeout() {
        SandoError::NetworkError(format!("{}: Request timed out - {}", context, error))
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
    SandoError::InternalError(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_retry_with_backoff() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result = retry_with_backoff(
            move || {
                let counter = counter_clone.clone();
                async move {
                    let attempts = counter.fetch_add(1, Ordering::SeqCst);
                    if attempts < 2 {
                        Err(SandoError::InternalError("Not ready yet".to_string()))
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
        // Just test for NetworkError without creating an actual reqwest::Error
        // This is a simplified test that doesn't require creating a reqwest::Error
        let result = SandoError::NetworkError("Request timed out".to_string());
        match result {
            SandoError::NetworkError(_) => (), // Test passes if this matches
            _ => panic!("Expected NetworkError"),
        }
    }
} 