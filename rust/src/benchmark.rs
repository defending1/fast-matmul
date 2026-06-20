use crate::matmul::MatMul;
use ndarray::Array2;
use rand::Rng;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// A struct for running benchmarks on various matrix multiplication algorithms.
pub struct Benchmark<'a> {
    matmul: &'a MatMul<'a>,
}

impl<'a> Benchmark<'a> {
    /// Creates a new `Benchmark` instance referencing a `MatMul` operator.
    pub fn new(matmul: &'a MatMul<'a>) -> Self {
        Self { matmul }
    }

    /// Computes C = A * B using classical matrix multiplication (using ndarray's built-in dot).
    pub fn classic_matmul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        a.dot(b)
    }

    /// Benchmarks classic_matmul, cp_matmul_single_thread, and cp_matmul,
    /// printing the elapsed times to the console and saving them to a CSV file.
    pub fn run(&self, sizes: &[usize], filename: &str) -> Result<(), std::io::Error> {
        if let Some(parent) = std::path::Path::new(filename).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = File::create(filename)?;
        writeln!(
            file,
            "size,classic_matmul,cp_matmul_single_thread,cp_matmul"
        )?;

        let mut rng = rand::thread_rng();

        for &size in sizes {
            println!("\nBenchmarking size {}x{}...", size, size);

            let mut a = Array2::zeros((size, size));
            let mut b = Array2::zeros((size, size));
            for val in a.iter_mut() {
                *val = rng.gen_range(-1.0..1.0);
            }
            for val in b.iter_mut() {
                *val = rng.gen_range(-1.0..1.0);
            }

            // 1. Classic MatMul
            let start = Instant::now();
            let _c_classic = self.classic_matmul(&a, &b);
            let duration_classic = start.elapsed().as_secs_f64();
            println!("  classic_matmul:          {:.6} s", duration_classic);

            // 2. CP MatMul Single-Thread
            let start = Instant::now();
            let _c_cp_single = self.matmul.cp_matmul_single_thread(&a, &b);
            let duration_cp_single = start.elapsed().as_secs_f64();
            println!("  cp_matmul_single_thread: {:.6} s", duration_cp_single);

            // 3. CP MatMul (Parallel/Multi-Thread)
            let start = Instant::now();
            let _c_cp_parallel = self.matmul.cp_matmul(&a, &b);
            let duration_cp_parallel = start.elapsed().as_secs_f64();
            println!("  cp_matmul:               {:.6} s", duration_cp_parallel);

            writeln!(
                file,
                "{},{},{},{}",
                size, duration_classic, duration_cp_single, duration_cp_parallel
            )?;
        }

        Ok(())
    }
}
