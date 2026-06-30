use fast_matmul::matmul::BaseMatMul;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Mapping from a benchmark name folder to its corresponding CSV header.
struct ColumnMapping {
    header: String,
    folder: String,
}

/// Helper to read a single point estimate of the mean from Criterion's JSON files, converting it to seconds.
fn get_criterion_time(folder_name: &str, size: usize) -> Option<f64> {
    let path = Path::new("target/criterion/Matrix Multiplication")
        .join(folder_name)
        .join(size.to_string())
        .join("new/estimates.json");
    if !path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    let nanoseconds = json.get("mean")?.get("point_estimate")?.as_f64()?;
    // Convert nanoseconds to seconds
    Some(nanoseconds / 1_000_000_000.0)
}

/// Reads existing benchmark CSV results to avoid overwriting unrelated cached data.
fn read_existing_csv(filename: &str) -> HashMap<(usize, String), f64> {
    let mut map = HashMap::new();
    let content = match std::fs::read_to_string(filename) {
        Ok(c) => c,
        Err(_) => return map,
    };
    let mut lines = content.lines();
    let header_line = match lines.next() {
        Some(h) => h,
        None => return map,
    };
    let headers: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    for line in lines {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.is_empty() || parts[0].trim().is_empty() {
            continue;
        }
        let size: usize = match parts[0].trim().parse() {
            Ok(s) => s,
            Err(_) => continue,
        };
        for (i, part) in parts.iter().enumerate().skip(1) {
            if let Some(val) = (i < headers.len())
                .then(|| part.trim().parse::<f64>().ok())
                .flatten()
            {
                map.insert((size, headers[i].clone()), val);
            }
        }
    }
    map
}

/// Exports Criterion results from target/criterion to a CSV file and runs the Python plot script.
/// Preserves existing data in the CSV if the benchmarks weren't re-run in the current session.
///
/// # Errors
///
/// Returns a standard `std::io::Error` if reading/writing file systems or calling Python script fails.
pub fn export_results_to_csv(
    sizes: &[usize],
    algorithms: &[&str],
    filename: &str,
    base_choice: BaseMatMul,
    plot: bool,
) -> Result<(), std::io::Error> {
    let existing = read_existing_csv(filename);

    let suffix = match base_choice {
        BaseMatMul::Faer => "Faer",
        BaseMatMul::Dgemm => "Dgemm",
    };

    let mut mappings = vec![
        ColumnMapping {
            header: "mkl_seq".to_string(),
            folder: "MKL-Sequential".to_string(),
        },
        ColumnMapping {
            header: "mkl_par".to_string(),
            folder: "MKL-Parallel".to_string(),
        },
        ColumnMapping {
            header: "faer_seq".to_string(),
            folder: "Faer-Sequential".to_string(),
        },
        ColumnMapping {
            header: "faer_par".to_string(),
            folder: "Faer-Parallel".to_string(),
        },
    ];

    for &algo in algorithms {
        let clean = algo.replace(['-', '.'], "_");
        mappings.push(ColumnMapping {
            header: format!("{}_seq", clean),
            folder: format!("{}-{}_Sequential", algo, suffix),
        });
        mappings.push(ColumnMapping {
            header: format!("{}_dfs", clean),
            folder: format!("{}-{}_DFS", algo, suffix),
        });
        mappings.push(ColumnMapping {
            header: format!("{}_bfs", clean),
            folder: format!("{}-{}_BFS", algo, suffix),
        });
        mappings.push(ColumnMapping {
            header: format!("{}_hybrid", clean),
            folder: format!("{}-{}_Hybrid", algo, suffix),
        });
    }

    // Only export sizes that have at least one valid measurement (either in Criterion files or existing CSV)
    let mut active_sizes = Vec::new();
    for &size in sizes {
        let mut has_data = false;
        for col in &mappings {
            if get_criterion_time(&col.folder, size).is_some()
                || existing.contains_key(&(size, col.header.clone()))
            {
                has_data = true;
                break;
            }
        }
        if has_data {
            active_sizes.push(size);
        }
    }

    if let Some(parent) = Path::new(filename).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(filename)?;

    write!(file, "size")?;
    for col in &mappings {
        write!(file, ",{}", col.header)?;
    }
    writeln!(file)?;

    for &size in &active_sizes {
        write!(file, "{}", size)?;
        for col in &mappings {
            let time_val = get_criterion_time(&col.folder, size)
                .or_else(|| existing.get(&(size, col.header.clone())).copied());

            if let Some(t) = time_val {
                write!(file, ",{:.9}", t)?;
            } else {
                write!(file, ",")?;
            }
        }
        writeln!(file)?;
    }

    println!("Successfully wrote benchmark CSV output to: {}", filename);

    if plot {
        let plot_script = if std::path::Path::new("python/plot.py").exists() {
            "python/plot.py"
        } else if std::path::Path::new("../python/plot.py").exists() {
            "../python/plot.py"
        } else {
            "python/plot.py"
        };

        println!(
            "Generating plot automatically using '{}' for '{}'...",
            plot_script, filename
        );

        let absolute_filename = std::path::Path::new(filename)
            .canonicalize()
            .unwrap_or_else(|_| std::path::PathBuf::from(filename));

        let plot_status = std::process::Command::new("uv")
            .args(["run", plot_script, &absolute_filename.to_string_lossy()])
            .status();
        match plot_status {
            Ok(status) if status.success() => {
                println!("Plot generated successfully.");
            }
            Ok(status) => {
                eprintln!("Plot generation failed with status: {:?}", status);
            }
            Err(e) => {
                eprintln!("Failed to execute plot script: {:?}", e);
            }
        }
    }

    Ok(())
}
