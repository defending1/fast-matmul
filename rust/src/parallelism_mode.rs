use std::convert::TryFrom;

/// The mode of parallel task execution to use in the fast matrix multiplication algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismMode {
    /// Depth-First Search (DFS) parallelism:
    /// Processes recursive steps sequentially and parallelizes inside the base/leaf GEMM calls.
    Dfs = 0,
    /// Breadth-First Search (BFS) parallelism:
    /// Spawns recursive tasks in parallel and runs the base/leaf GEMM calls sequentially.
    Bfs = 1,
    /// Hybrid parallelism:
    /// Spawns recursive tasks in parallel at the top levels, and switches to DFS style (sequential tasks
    /// with multithreaded GEMM) at the lower levels once thread capacity is saturated.
    Hybrid = 2,
    /// Single-threaded execution (completely sequential).
    Sequential = 3,
}

impl TryFrom<i32> for ParallelismMode {
    type Error = String;

    /// Attempts to convert an `i32` value to a `ParallelismMode`.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not in the range `0..=3`.
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ParallelismMode::Dfs),
            1 => Ok(ParallelismMode::Bfs),
            2 => Ok(ParallelismMode::Hybrid),
            3 => Ok(ParallelismMode::Sequential),
            _ => Err(format!(
                "Invalid parallelism mode: {}. Must be 0 (Dfs), 1 (Bfs), 2 (Hybrid), or 3 (Sequential).",
                value
            )),
        }
    }
}

/// The base matrix multiplication implementation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseMatMul {
    /// Use classical matrix multiplication from the `faer` library.
    Faer = 0,
    /// Use Intel MKL `dgemm` (via FFI wrappers).
    Dgemm = 1,
}

impl TryFrom<i32> for BaseMatMul {
    type Error = String;

    /// Attempts to convert an `i32` value to a `BaseMatMul`.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not `0` or `1`.
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BaseMatMul::Faer),
            1 => Ok(BaseMatMul::Dgemm),
            _ => Err(format!(
                "Invalid base matmul choice: {}. Must be 0 (Faer) or 1 (Dgemm).",
                value
            )),
        }
    }
}
