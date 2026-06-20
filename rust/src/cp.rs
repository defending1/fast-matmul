use ndarray::Array2;
use std::fs;
use std::sync::OnceLock;

static STRASSEN_CP: OnceLock<CP> = OnceLock::new();

/// Canonical Polyadic (CP) decomposition matrices [U, V, W] for matrix multiplication.
#[derive(Clone, Debug)]
pub struct CP {
    pub u: Array2<f64>,
    pub v: Array2<f64>,
    pub w: Array2<f64>,
    pub m: usize,
    pub n: usize,
    pub p: usize,
    pub rank: usize,
}

impl CP {
    /// Loads a CP decomposition from a file by name.
    /// Rejects approximation algorithms containing `approx`.
    /// Rejects subexpression-eliminated algorithms containing `Substitution information`.
    /// Derives dimensions (M, N, P) mathematically and validates the rank.
    pub fn load(name: &str) -> Self {
        assert!(
            !name.contains("approx"),
            "Algorithm '{}' is an approximation algorithm, which is ignored.",
            name
        );

        let paths = [
            "codegen/algorithms",
            "../codegen/algorithms",
            "../../codegen/algorithms",
        ];
        let mut content = None;
        let mut resolved_path = None;
        for base in &paths {
            let path = format!("{}/{}", base, name);
            if let Ok(c) = fs::read_to_string(&path) {
                content = Some(c);
                resolved_path = Some(path);
                break;
            }
        }
        let content = content.unwrap_or_else(|| {
            panic!(
                "Could not locate algorithm file '{}'. Make sure to run from the project root or rust directory.",
                name
            )
        });
        let path_str = resolved_path.unwrap();

        assert!(
            !content.contains("Substitution information"),
            "Algorithm '{}' contains subexpression substitution/elimination information, which is ignored.",
            name
        );

        let mut matrices = Vec::new();
        let mut current_rows: Vec<Vec<f64>> = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with('#') {
                if !current_rows.is_empty() {
                    let num_rows = current_rows.len();
                    let num_cols = current_rows[0].len();
                    let flat: Vec<f64> = current_rows.into_iter().flatten().collect();
                    matrices.push(
                        Array2::from_shape_vec((num_rows, num_cols), flat)
                            .expect("Invalid matrix shape"),
                    );
                    current_rows = Vec::new();
                }
                continue;
            }

            let row: Vec<f64> = trimmed
                .split_whitespace()
                .map(parse_float_with_fraction)
                .collect();
            current_rows.push(row);
        }

        if !current_rows.is_empty() {
            let num_rows = current_rows.len();
            let num_cols = current_rows[0].len();
            let flat: Vec<f64> = current_rows.into_iter().flatten().collect();
            matrices.push(
                Array2::from_shape_vec((num_rows, num_cols), flat).expect("Invalid matrix shape"),
            );
        }

        assert_eq!(
            matrices.len(),
            3,
            "Expected exactly 3 matrices in the CP decomposition file"
        );

        let u = matrices[0].clone();
        let v = matrices[1].clone();
        let w = matrices[2].clone();

        let r_u = u.nrows();
        let r_v = v.nrows();
        let r_w = w.nrows();

        let rank = u.ncols();
        assert_eq!(
            v.ncols(),
            rank,
            "Matrix V columns must match U columns (rank)"
        );
        assert_eq!(
            w.ncols(),
            rank,
            "Matrix W columns must match U columns (rank)"
        );

        // Derive dimensions M, N, P mathematically:
        // M^2 = (R_U * R_W) / R_V
        let m2 = (r_u * r_w) / r_v;
        let m = (m2 as f64).sqrt().round() as usize;
        assert_eq!(m * m, m2, "Derivation of M failed: not a perfect square");
        let n = r_u / m;
        assert_eq!(m * n, r_u, "Derivation of N failed");
        let p = r_w / m;
        assert_eq!(m * p, r_w, "Derivation of P failed");
        assert_eq!(n * p, r_v, "Derivation of N * P = R_V failed");

        // Verify the rank claim in the filename if applicable
        let filename = std::path::Path::new(&path_str)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("");

        if let Some(expected_rank) = parse_rank_from_filename(filename) {
            assert_eq!(
                rank, expected_rank,
                "Rank mismatch for algorithm '{}': filename specifies {}, but file contains {}",
                name, expected_rank, rank
            );
        } else {
            println!(
                "Rank not defined in filename for '{}'. Computed rank from matrix columns: {}",
                name, rank
            );
        }

        CP {
            u,
            v,
            w,
            m,
            n,
            p,
            rank,
        }
    }

    /// Computes and returns the rank of this CP decomposition (number of components).
    pub fn compute_rank(&self) -> usize {
        self.u.ncols()
    }

    /// Returns a reference to the statically loaded Strassen CP decomposition matrices.
    pub fn get_strassen() -> &'static Self {
        STRASSEN_CP.get_or_init(|| Self::load("strassen"))
    }
}

/// Parses float inputs that may contain fractional slash representation (e.g. "1/8", "-1/8").
fn parse_float_with_fraction(s: &str) -> f64 {
    if let Some(pos) = s.find('/') {
        let num = s[..pos].parse::<f64>().expect("Invalid numerator");
        let den = s[pos + 1..].parse::<f64>().expect("Invalid denominator");
        num / den
    } else {
        s.parse::<f64>().expect("Invalid float")
    }
}

/// Helper function to parse the rank from the filename if it matches the pattern name-rank-additions
fn parse_rank_from_filename(filename: &str) -> Option<usize> {
    let parts: Vec<&str> = filename.split('-').collect();
    if parts.len() == 3 {
        parts[1].parse::<usize>().ok()
    } else {
        None
    }
}
