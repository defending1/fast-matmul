use crate::cp::CP;
use crate::matmul::MatMul;
use faer::Mat;
use rand::Rng;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// A struct for running benchmarks on various matrix multiplication algorithms.
pub struct Benchmark;

impl Default for Benchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl Benchmark {
    /// Creates a new `Benchmark` instance.
    pub fn new() -> Self {
        Self
    }

    /// Computes C = A * B using classical matrix multiplication (using faer *).
    pub fn classic_matmul(&self, a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
        a * b
    }

    /// Benchmarks classic_matmul and the CP matmul (single/multi-thread) for multiple algorithms,
    /// printing elapsed times to the console and saving results to a CSV file.
    pub fn run(
        &self,
        sizes: &[usize],
        algorithms: &[&str],
        filename: &str,
    ) -> Result<(), std::io::Error> {
        if let Some(parent) = std::path::Path::new(filename).parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Load CP decompositions for all requested algorithms
        let mut cps = Vec::new();
        for &algo in algorithms {
            println!("Loading CP decomposition for '{}'...", algo);
            let cp = CP::load(algo);
            cps.push((algo, cp));
        }

        let mut file = File::create(filename)?;

        // Write header: size,system,mkl,algo1_single,algo1_multithread,algo2_single,...
        write!(file, "size,system,mkl")?;
        for &(algo, _) in &cps {
            let clean_name = algo.replace('-', "_").replace('.', "_");
            write!(file, ",{}_single,{}_multithread", clean_name, clean_name)?;
        }
        writeln!(file)?;

        let mut rng = rand::thread_rng();

        for &size in sizes {
            println!("\nBenchmarking size {}x{}...", size, size);

            let mut a = Mat::<f64>::zeros(size, size);
            let mut b = Mat::<f64>::zeros(size, size);
            for r in 0..size {
                for c in 0..size {
                    a[(r, c)] = rng.gen_range(-1.0..1.0);
                    b[(r, c)] = rng.gen_range(-1.0..1.0);
                }
            }

            // 1. Classic/System MatMul
            let start = Instant::now();
            let _c_classic = self.classic_matmul(&a, &b);
            let duration_classic = start.elapsed().as_secs_f64();
            println!("  system:                  {:.6} s", duration_classic);

            // 1b. MKL MatMul
            let start = Instant::now();
            let _c_mkl = crate::mkl::mkl_matmul(&a, &b);
            let duration_mkl = start.elapsed().as_secs_f64();
            println!("  mkl:                     {:.6} s", duration_mkl);

            // Verification of MKL against System/Classic
            let mut max_diff = 0.0f64;
            for r in 0..size {
                for c in 0..size {
                    let diff = (_c_classic[(r, c)] - _c_mkl[(r, c)]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
            }
            if max_diff > 1e-12 {
                println!(
                    "  WARNING: MKL result differs from system by max diff {}",
                    max_diff
                );
            }

            write!(file, "{},{},{}", size, duration_classic, duration_mkl)?;

            // 2. CP MatMul for each algorithm
            for &(algo, ref cp) in &cps {
                let mm = MatMul::with_cp(cp);

                // Single-thread
                let start = Instant::now();
                let _c_cp_single = mm.cp_matmul_single_thread(&a, &b);
                let duration_cp_single = start.elapsed().as_secs_f64();
                println!("  {}_single: {:.6} s", algo, duration_cp_single);

                // Multi-thread
                let start = Instant::now();
                let _c_cp_parallel = mm.cp_matmul(&a, &b);
                let duration_cp_parallel = start.elapsed().as_secs_f64();
                println!("  {}_multithread: {:.6} s", algo, duration_cp_parallel);

                write!(file, ",{},{}", duration_cp_single, duration_cp_parallel)?;
            }
            writeln!(file)?;
        }

        Ok(())
    }
}
