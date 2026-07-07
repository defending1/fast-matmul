use fast_matmul::matmul::{BaseMatMul, RecursionLimit};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Mapping from a benchmark name folder to its corresponding CSV header.
struct ColumnMapping {
    header: String,
    folder: String,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct ConfigKey {
    size: usize,
    base_choice: String,
    recursion_level: Option<usize>,
    size_cutoff: Option<usize>,
}

impl Ord for ConfigKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.size.cmp(&other.size)
            .then_with(|| self.base_choice.cmp(&other.base_choice))
            .then_with(|| self.recursion_level.cmp(&other.recursion_level))
            .then_with(|| self.size_cutoff.cmp(&other.size_cutoff))
    }
}

impl PartialOrd for ConfigKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}



/// Reads existing benchmark CSV results to avoid overwriting unrelated cached data.
fn read_existing_csv(filename: &str) -> HashMap<ConfigKey, HashMap<String, f64>> {
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

    let size_idx = headers.iter().position(|h| h == "size");
    let base_choice_idx = headers.iter().position(|h| h == "base_choice");
    let recursion_level_idx = headers.iter().position(|h| h == "recursion_level");
    let size_cutoff_idx = headers.iter().position(|h| h == "size_cutoff");

    let (size_idx, base_choice_idx, recursion_level_idx, size_cutoff_idx) = match (
        size_idx,
        base_choice_idx,
        recursion_level_idx,
        size_cutoff_idx,
    ) {
        (Some(s), Some(b), Some(r), Some(c)) => (s, b, r, c),
        _ => return map,
    };

    for line in lines {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() <= size_cutoff_idx {
            continue;
        }
        let size: usize = match parts[size_idx].trim().parse() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let base_choice = parts[base_choice_idx].trim().to_string();
        let recursion_level = parts[recursion_level_idx].trim().parse::<usize>().ok();
        let size_cutoff = parts[size_cutoff_idx].trim().parse::<usize>().ok();

        let key = ConfigKey {
            size,
            base_choice,
            recursion_level,
            size_cutoff,
        };

        let mut row_metrics = HashMap::new();
        for (i, part) in parts.iter().enumerate() {
            if i == size_idx || i == base_choice_idx || i == recursion_level_idx || i == size_cutoff_idx {
                continue;
            }
            if i < headers.len() {
                let cleaned = part.trim();
                if let Ok(val) = cleaned.parse::<f64>() {
                    row_metrics.insert(headers[i].clone(), val);
                }
            }
        }
        if !row_metrics.is_empty() {
            map.insert(key, row_metrics);
        }
    }
    map
}

/// Exports benchmark results from in-memory timings to a CSV file and runs the Python plot script.
/// Preserves existing data in the CSV if the benchmarks weren't re-run in the current session.
///
/// # Arguments
/// * `sizes` - A slice of matrix dimension sizes.
/// * `algorithms` - A slice of algorithm names.
/// * `filename` - The output CSV filename.
/// * `base_choice` - The base matrix multiplication choice.
/// * `recursion_limit` - The recursion limit choice.
/// * `plot` - Whether to generate a performance plot from the CSV.
/// * `new_timings` - In-memory map of benchmark names to map of size and elapsed time.
///
/// # Errors
///
/// Returns a standard `std::io::Error` if reading/writing file systems or calling Python script fails.
pub fn export_results_to_csv(
    sizes: &[usize],
    algorithms: &[&str],
    filename: &str,
    base_choice: BaseMatMul,
    recursion_limit: RecursionLimit,
    plot: bool,
    new_timings: &HashMap<String, HashMap<usize, f64>>,
) -> Result<(), std::io::Error> {
    let mut existing = read_existing_csv(filename);

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

    let config_suffix = match recursion_limit {
        RecursionLimit::Depth(level) => format!("level_{}", level),
        RecursionLimit::Cutoff(cutoff) => format!("cutoff_{}", cutoff),
    };

    for &algo in algorithms {
        let clean = algo.replace(['-', '.'], "_");
        mappings.push(ColumnMapping {
            header: format!("{}_seq", clean),
            folder: format!("{}-{}-{}_Sequential", algo, suffix, config_suffix),
        });
        mappings.push(ColumnMapping {
            header: format!("{}_dfs", clean),
            folder: format!("{}-{}-{}_DFS", algo, suffix, config_suffix),
        });
        mappings.push(ColumnMapping {
            header: format!("{}_bfs", clean),
            folder: format!("{}-{}-{}_BFS", algo, suffix, config_suffix),
        });
        mappings.push(ColumnMapping {
            header: format!("{}_hybrid", clean),
            folder: format!("{}-{}-{}_Hybrid", algo, suffix, config_suffix),
        });
    }

    let base_choice_str = match base_choice {
        BaseMatMul::Faer => "faer",
        BaseMatMul::Dgemm => "dgemm",
    };

    let (recursion_level, size_cutoff) = match recursion_limit {
        RecursionLimit::Depth(level) => (Some(level), None),
        RecursionLimit::Cutoff(cutoff) => (None, Some(cutoff)),
    };

    // 1. Gather new measurements, merging with existing data
    for &size in sizes {
        let mut has_new_data = false;
        let mut new_metrics = HashMap::new();

        for col in &mappings {
            if let Some(t) = new_timings.get(&col.folder).and_then(|m| m.get(&size).copied()) {
                new_metrics.insert(col.header.clone(), t);
                has_new_data = true;
            }
        }

        if has_new_data {
            let key = ConfigKey {
                size,
                base_choice: base_choice_str.to_string(),
                recursion_level,
                size_cutoff,
            };
            let row = existing.entry(key).or_default();
            for (header, val) in new_metrics {
                row.insert(header, val);
            }
        }
    }

    // 1.5. Propagate baseline measurements to all other configurations of the same size
    let base_headers = ["mkl_seq", "mkl_par", "faer_seq", "faer_par"];
    for &size in sizes {
        let mut size_base_timings = HashMap::new();
        for &header in &base_headers {
            let folder = match header {
                "mkl_seq" => "MKL-Sequential",
                "mkl_par" => "MKL-Parallel",
                "faer_seq" => "Faer-Sequential",
                "faer_par" => "Faer-Parallel",
                _ => unreachable!(),
            };
            if let Some(t) = new_timings.get(folder).and_then(|m| m.get(&size).copied()) {
                size_base_timings.insert(header.to_string(), t);
            }
        }

        if !size_base_timings.is_empty() {
            for (k, row_metrics) in &mut existing {
                if k.size == size {
                    for (header, &t) in &size_base_timings {
                        row_metrics.insert(header.clone(), t);
                    }
                }
            }
        }
    }

    // 2. Write everything back to CSV
    if let Some(parent) = Path::new(filename).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(filename)?;

    write!(file, "size,base_choice,recursion_level,size_cutoff")?;
    for col in &mappings {
        write!(file, ",{}", col.header)?;
    }
    writeln!(file)?;

    let mut sorted_keys: Vec<&ConfigKey> = existing.keys().collect();
    sorted_keys.sort();

    for key in sorted_keys {
        let row_metrics = &existing[key];
        if row_metrics.is_empty() {
            continue;
        }

        write!(file, "{},{}", key.size, key.base_choice)?;
        if let Some(level) = key.recursion_level {
            write!(file, ",{}", level)?;
        } else {
            write!(file, ",")?;
        }
        if let Some(cutoff) = key.size_cutoff {
            write!(file, ",{}", cutoff)?;
        } else {
            write!(file, ",")?;
        }

        for col in &mappings {
            if let Some(t) = row_metrics.get(&col.header) {
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

