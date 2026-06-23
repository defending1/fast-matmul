use fast_matmul::benchmark::Benchmark;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let plot_only = args.iter().any(|arg| arg == "--plot-only" || arg == "-p");

    let full = args.iter().any(|arg| arg == "--full");

    // Default limit is 2^10 (1024). Under --full, we run up to 2^20 (1,048,576),
    // which will dynamically check system memory and stop before exceeding limits.
    let n_limit = if full { 20 } else { 10 };
    let sizes: Vec<usize> = (1..=n_limit).map(|n| 1usize << n).collect(); // 2, 4, ..., 2^N
    let csv_file = "generated/benchmark_results.csv";
    let algorithms = &["strassen", "grey-strassen"];

    if plot_only {
        println!("Plot-only mode: Regenerating CSV results from cached Criterion data...");
        if let Err(e) = Benchmark::export_results_to_csv(&sizes, algorithms, csv_file) {
            eprintln!("Failed to export CSV: {:?}", e);
        } else {
            println!("CSV results successfully updated from cache.");
        }
    } else {
        println!("\n--- Running Matrix Multiplication Benchmarks ---");
        let bench = Benchmark::new();
        if let Err(e) = bench.run(&sizes, algorithms, csv_file) {
            eprintln!("Failed to write benchmarks to CSV: {:?}", e);
        } else {
            println!("Benchmark results successfully written to {}", csv_file);
        }
    }
}
