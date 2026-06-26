mod base_matmul;

use faer::Mat;
use fast_matmul::matmul::BaseMatMul;
use rand::Rng;
use std::io::Write;

/// Helper function to generate a random matrix of double precision floats.
fn random_matrix(rows: usize, cols: usize) -> Mat<f64> {
    let mut rng = rand::thread_rng();
    Mat::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0))
}

/// Helper function to time an execution and return duration in milliseconds.
fn time_fn<F>(mut func: F) -> f64
where
    F: FnMut(),
{
    let t1 = std::time::Instant::now();
    func();
    let duration = t1.elapsed();
    duration.as_secs_f64() * 1000.0
}



/// Run a single timing test and output in the same format as the C++ benchmark.
fn run_single_test(a: &Mat<f64>, b: &Mat<f64>, multithreaded: bool) {
    let num_trials = 5;
    let time = time_fn(|| {
        for _ in 0..num_trials {
            let _c = base_matmul::base_matmul(a, b, multithreaded, BaseMatMul::Faer);
        }
    });
    print!(
        " {} {} {} {} {:.6};",
        a.nrows(),
        a.ncols(),
        b.ncols(),
        num_trials,
        time
    );
    std::io::stdout().flush().unwrap();
}

fn main() {
    let positional_args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with('-'))
        .collect();

    if positional_args.is_empty() {
        println!("Usage: cargo bench --bench dgemm_curves -- <TYPE> [NUM_THREADS]");
        println!("  TYPE: 1 (SQUARE), 2 (OUTER_PRODUCT), 3 (TS_SQUARE)");
        println!(
            "  NUM_THREADS: number of threads (1 for sequential, 0 or omitted for maximum parallel)"
        );
        return;
    }

    let type_opt: i32 = positional_args[0]
        .parse()
        .expect("Invalid type option (must be 1, 2, or 3)");
    let num_threads: i32 = if positional_args.len() > 1 {
        positional_args[1]
            .parse()
            .expect("Invalid thread count option")
    } else {
        0
    };

    // Configure threads if needed (faer uses Rayon thread pool internally by default)
    let multithreaded = num_threads != 1;
    if num_threads > 1 {
        // Build thread pool with requested threads if it doesn't already exist or configure rayon
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build_global();
    }

    let n_vals: Vec<usize> = if !multithreaded {
        (25..=3000).step_by(25).collect()
    } else {
        (200..=8000).step_by(100).collect()
    };

    for &n in &n_vals {
        let (a, b) = match type_opt {
            1 => {
                // SQUARE: N x N x N
                (random_matrix(n, n), random_matrix(n, n))
            }
            2 => {
                // OUTER_PRODUCT: N x 800 x N
                (random_matrix(n, 800), random_matrix(800, n))
            }
            3 => {
                // TS_SQUARE: N x 800 x 800
                (random_matrix(n, 800), random_matrix(800, 800))
            }
            _ => panic!("Incorrect type option (must be 1, 2, or 3)"),
        };

        run_single_test(&a, &b, multithreaded);
        if n % 1000 == 0 {
            println!("...");
        }
    }
    println!();
}
