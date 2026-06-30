/// Helper to validate memory allocation limits and check system memory capacity for benchmarks.
pub struct CheckMemoryHelper;

impl CheckMemoryHelper {
    /// Checks if the matrix size is supported by the machine's memory and limits.
    ///
    /// Returns `Ok(())` if the size is supported, or an `Err(String)` containing
    /// a descriptive message of why it is not supported.
    pub fn check_size_supported(size: usize) -> Result<(), String> {
        if size == 0 {
            return Err("Matrix size must be greater than 0.".to_string());
        }

        // 1. Check for arithmetic overflow in size calculations
        let elements = size.checked_mul(size).ok_or_else(|| {
            format!(
                "Matrix size {}x{} would overflow usize elements count.",
                size, size
            )
        })?;

        let bytes_per_matrix = elements
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                format!(
                    "Matrix size {}x{} would overflow memory byte count.",
                    size, size
                )
            })?;

        // Rust's allocator limit is isize::MAX
        if bytes_per_matrix > isize::MAX as usize {
            return Err(format!(
                "Matrix size {}x{} requires {} bytes, which exceeds Rust's maximum allocation limit of {} bytes.",
                size,
                size,
                bytes_per_matrix,
                isize::MAX
            ));
        }

        // Estimate total peak memory required for the benchmark at this size.
        // We run multiple algorithms (MKL, Faer, Strassen single/multi-thread).
        // Strassen in parallel mode with Rayon has the highest peak memory overhead.
        // Let's estimate peak memory overhead as:
        // Input matrices (A, B) + output matrix (C) + concurrent workspace.
        // A safe factor for Strassen parallel is (3 + T * 1.5) where T is the thread count.
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let multiplier = 3.0 + (num_threads as f64).min(7.0) * 1.5;
        let estimated_required_bytes = (bytes_per_matrix as f64 * multiplier) as u64;

        if let Some(avail_bytes) = Self::get_available_memory() {
            // Keep a safety buffer: 10% of available memory or at least 256MB free
            let safety_buffer = (avail_bytes / 10).max(256 * 1024 * 1024);
            if estimated_required_bytes + safety_buffer > avail_bytes {
                return Err(format!(
                    "Matrix size {}x{} requires estimated {} MB of memory (with safety buffer), but only {} MB is available.",
                    size,
                    size,
                    (estimated_required_bytes + safety_buffer) / (1024 * 1024),
                    avail_bytes / (1024 * 1024)
                ));
            }
        }

        Ok(())
    }

    /// Helper function to parse `/proc/meminfo` and return the available memory in bytes.
    /// If not on Linux or if it fails, returns `None`.
    fn get_available_memory() -> Option<u64> {
        let content = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in content.lines() {
            if line.to_ascii_lowercase().starts_with("memavailable:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(kb) = parts.get(1).and_then(|s| s.parse::<u64>().ok()) {
                    return Some(kb * 1024); // Convert kB to bytes
                }
            }
        }
        None
    }
}
