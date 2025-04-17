use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

/// Maximum log file size in bytes before rotation is suggested (50MB)
const MAX_LOG_SIZE: u64 = 50 * 1024 * 1024;

/// Maximum number of log files to keep (7 days)
const MAX_LOG_FILES: usize = 7;

/// Checks log directory and reports on file sizes and growth
pub fn check_log_directory(log_dir: &str) -> anyhow::Result<()> {
    let log_path = Path::new(log_dir);
    if !log_path.exists() {
        fs::create_dir_all(log_path)?;
        info!(target: "log_management", "Created log directory: {}", log_dir);
        return Ok(());
    }
    
    // Get all log files in the directory
    let entries = fs::read_dir(log_path)?;
    let mut log_files = Vec::new();
    
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && 
           path.extension().map_or(false, |ext| ext == "log") || 
           path.to_string_lossy().contains("sandoseer.log") {
            
            if let Ok(metadata) = fs::metadata(&path) {
                log_files.push((path, metadata.len()));
            }
        }
    }
    
    // Calculate total size of log files
    let total_size: u64 = log_files.iter().map(|(_, size)| size).sum();
    
    // Warn if total size is large
    if total_size > MAX_LOG_SIZE {
        warn!(
            target: "log_management",
            total_size_mb = total_size / (1024 * 1024),
            max_size_mb = MAX_LOG_SIZE / (1024 * 1024),
            "Log directory size exceeds recommended maximum"
        );
    } else {
        info!(
            target: "log_management",
            total_size_mb = total_size / (1024 * 1024),
            log_count = log_files.len(),
            "Log directory size within limits"
        );
    }
    
    // Print detailed information about each log file
    for (path, size) in &log_files {
        info!(
            target: "log_management", 
            path = %path.display(),
            size_mb = size / (1024 * 1024),
            "Log file info"
        );
    }
    
    Ok(())
}

/// Rotate logs by removing old files if there are too many
pub fn rotate_logs(log_dir: &str) -> anyhow::Result<()> {
    let log_path = Path::new(log_dir);
    if !log_path.exists() {
        return Ok(());
    }
    
    // Get all log files in the directory
    let entries = fs::read_dir(log_path)?;
    let mut log_files = Vec::new();
    
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && 
           path.extension().map_or(false, |ext| ext == "log") || 
           path.to_string_lossy().contains("sandoseer.log") {
            
            if let Ok(metadata) = fs::metadata(&path) {
                let modified = metadata.modified()?
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                log_files.push((path, modified));
            }
        }
    }
    
    // Sort by modification time (newest first)
    log_files.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Remove old files if there are too many
    if log_files.len() > MAX_LOG_FILES {
        for (path, _) in log_files.iter().skip(MAX_LOG_FILES) {
            info!(target: "log_management", path = %path.display(), "Removing old log file");
            if let Err(e) = fs::remove_file(path) {
                warn!(target: "log_management", path = %path.display(), error = %e, "Failed to remove old log file");
            }
        }
    }
    
    Ok(())
}

/// Clear old debug logs
pub fn clear_debug_logs(log_dir: &str) -> anyhow::Result<()> {
    let log_path = Path::new(log_dir);
    if !log_path.exists() {
        return Ok(());
    }
    
    // Get all debug log files
    let entries = fs::read_dir(log_path)?;
    
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        // Only clear files with .debug extension
        if path.is_file() && path.extension().map_or(false, |ext| ext == "debug") {
            info!(target: "log_management", path = %path.display(), "Removing debug log file");
            if let Err(e) = fs::remove_file(&path) {
                warn!(target: "log_management", path = %path.display(), error = %e, "Failed to remove debug log file");
            }
        }
    }
    
    Ok(())
} 