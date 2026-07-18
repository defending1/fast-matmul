use faer::Mat;
use fast_matmul::matmul::BaseMatMul;
use rand::Rng;
use splinefit::CubicSplineFit;
use spliny::SplineCurve;
use std::io::Write;

/// Runs a benchmark function using minstant with warmup and exactly 2 samples.
///
/// # Arguments
/// * `warmup_ms` - The duration in milliseconds to run warmup iterations.
/// * `measure_ms` - The target measurement time in milliseconds for the samples.
/// * `f` - The benchmark function to execute.
pub fn run_benchmark_minstant<F, O>(
    _warmup_ms: u64,
    _measure_ms: u64,
    mut f: F,
) -> f64
where
    F: FnMut() -> O,
{
    let warmup_duration = std::time::Duration::from_millis(2);
    let warmup_start = minstant::Instant::now();
    while warmup_start.elapsed() < warmup_duration {
        let _ = f();
    }

    let start = minstant::Instant::now();
    let _ = f();
    start.elapsed().as_secs_f64()
}

#[allow(dead_code)]
unsafe extern "C" {
    fn spalder_(
        t: *const f64,
        n: *const i32,
        c: *const f64,
        k: *const i32,
        nu: *const i32,
        x: *const f64,
        y: *mut f64,
        m: *const i32,
        wrk: *mut f64,
        ier: *mut i32,
    );
}

/// Run the baseline sequential or parallel matrix multiplication,
/// selectively using `faer` or Intel MKL `dgemm` based on the specified choice.
pub fn base_matmul(
    a: &Mat<f64>,
    b: &Mat<f64>,
    multithreaded: bool,
    base_choice: BaseMatMul,
) -> Mat<f64> {
    fast_matmul::matmul::init_rayon_threads();
    match base_choice {
        BaseMatMul::Faer => {
            let mut c = Mat::zeros(a.nrows(), b.ncols());
            let par = if multithreaded {
                faer::get_global_parallelism()
            } else {
                faer::Par::Seq
            };
            faer::linalg::matmul::matmul(
                c.as_mut(),
                faer::Accum::Replace,
                a.as_ref(),
                b.as_ref(),
                1.0,
                par,
            );
            c
        }
        BaseMatMul::Dgemm => {
            // Adjust thread count for MKL dynamically based on concurrency requirements.
            // Single-threaded GEMM runs sequentially; multithreaded GEMM uses all available cores.
            fast_matmul::mkl::mkl_set_threads(if multithreaded { 0 } else { 1 });
            fast_matmul::mkl::mkl_matmul(a, b)
        }
    }
}

/// Run the baseline sequential or parallel matrix multiplication in-place,
/// selectively using `faer` or Intel MKL `dgemm` based on the specified choice.
#[allow(dead_code)]
pub fn base_matmul_inplace(
    dst: faer::MatMut<'_, f64>,
    a: faer::MatRef<'_, f64>,
    b: faer::MatRef<'_, f64>,
    multithreaded: bool,
    base_choice: BaseMatMul,
) {
    fast_matmul::matmul::init_rayon_threads();
    match base_choice {
        BaseMatMul::Faer => {
            let par = if multithreaded {
                faer::get_global_parallelism()
            } else {
                faer::Par::Seq
            };
            faer::linalg::matmul::matmul(
                dst,
                faer::Accum::Replace,
                a,
                b,
                1.0,
                par,
            );
        }
        BaseMatMul::Dgemm => {
            fast_matmul::mkl::mkl_set_threads(if multithreaded { 0 } else { 1 });
            fast_matmul::mkl::mkl_matmul_impl(dst, faer::Accum::Replace, a, b);
        }
    }
}


/// Evaluates the first derivative of a 1D spline at the specified query points.
#[allow(dead_code)]
pub fn evaluate_spline_derivative<const K: usize>(
    s: &SplineCurve<K, 1>,
    x: &[f64],
) -> std::result::Result<Vec<f64>, i32> {
    let n = s.t.len() as i32;
    let k = K as i32;
    let nu = 1i32; // first derivative
    let m = x.len() as i32;
    let mut y = vec![0.0; x.len()];
    let mut ier = 0i32;

    // Pad coefficients as required by spalder_
    let mut c_full = vec![0.0f64; s.t.len()];
    c_full[..s.c.len()].copy_from_slice(&s.c);

    let mut wrk = vec![0.0f64; s.t.len()];

    unsafe {
        spalder_(
            s.t.as_ptr(),
            &n,
            c_full.as_ptr(),
            &k,
            &nu,
            x.as_ptr(),
            y.as_mut_ptr(),
            &m,
            wrk.as_mut_ptr(),
            &mut ier,
        );
    }

    if ier == 0 { Ok(y) } else { Err(ier) }
}

fn random_matrix(rows: usize, cols: usize) -> Mat<f64> {
    let mut rng = rand::thread_rng();
    Mat::from_fn(rows, cols, |_, _| rng.gen_range(-1.0..1.0))
}

/// Runs baseline sequential matrix multiplication for the given sizes,
/// records execution times, calculates effective GFLOPS, fits an interpolating spline,
/// computes the spline derivative (dGFLOPS/dN) at those sizes, exports results to a CSV file,
/// and returns the GFLOPS and derivatives.
#[allow(dead_code)]
pub fn fit_and_differentiate_spline(
    csv_path: &str,
    n_vals: &[f64],
) -> std::result::Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let mut times = Vec::new();

    println!("Measuring execution times for specified N values...");

    for &n in n_vals {
        let size = n as usize;
        let a = random_matrix(size, size);
        let b = random_matrix(size, size);

        let warmup_duration = std::time::Duration::from_millis(2);
        let warmup_start = std::time::Instant::now();
        while warmup_start.elapsed() < warmup_duration {
            let _ = base_matmul(&a, &b, false, BaseMatMul::Faer);
        }

        let start = std::time::Instant::now();
        let _ = base_matmul(&a, &b, false, BaseMatMul::Faer);
        let duration_s = start.elapsed().as_secs_f64();
        times.push(duration_s);
    }

    // Calculate effective GFLOPS: (2n^3 - n^2) / (time_s * 1e9)
    let mut gflops = Vec::new();
    for i in 0..n_vals.len() {
        let n: f64 = n_vals[i];
        let time_s = times[i];
        let flops = 2.0 * n * n * n - n * n;
        let gflops_val = flops / (time_s * 1e9);
        gflops.push(gflops_val);
    }

    // Fit cubic interpolating spline on GFLOPS
    let spline = CubicSplineFit::new(n_vals.to_vec(), gflops.clone()).interpolating_spline()?;

    // Compute derivative of GFLOPS spline at points
    let derivatives = evaluate_spline_derivative(&spline, n_vals)
        .map_err(|ier| format!("Dierckx spalder error: {}", ier))?;

    // Create containing folder if it does not exist
    if let Some(parent) = std::path::Path::new(csv_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Write to CSV
    let mut file = std::fs::File::create(csv_path)?;
    writeln!(file, "size,gflops,derivative")?;
    for i in 0..n_vals.len() {
        writeln!(file, "{},{},{}", n_vals[i], gflops[i], derivatives[i])?;
    }
    println!("Successfully wrote spline CSV output to: {}", csv_path);

    Ok((gflops, derivatives))
}

/// Helper function to dynamically find the project root directory.
/// Resolves root by checking if the CWD or any of its parents contain the `rust` subdirectory with a Cargo.toml.
pub fn get_project_root() -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap_or_default();
    if cwd.join("rust").join("Cargo.toml").exists() {
        cwd
    } else if cwd.file_name().is_some_and(|n| n == "rust") {
        cwd.parent().map(|p| p.to_path_buf()).unwrap_or(cwd)
    } else {
        let mut dir = cwd.clone();
        while dir.pop() {
            if dir.join("rust").join("Cargo.toml").exists() {
                return dir;
            }
        }
        cwd
    }
}


