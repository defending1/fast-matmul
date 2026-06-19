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
}

impl CP {
    /// Loads Strassen CP decomposition matrices [U, V, W] from the file `codegen/algorithms/strassen`.
    /// If loading fails, it panics with a descriptive message.
    pub fn load_strassen() -> Self {
        let paths = [
            "codegen/algorithms/strassen",
            "../codegen/algorithms/strassen",
            "../../codegen/algorithms/strassen",
        ];
        let mut content = None;
        for path in &paths {
            if let Ok(c) = fs::read_to_string(path) {
                content = Some(c);
                break;
            }
        }
        let content = content.expect(
            "Could not locate 'codegen/algorithms/strassen'. Make sure to run from the project root or rust directory."
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
                .map(|s| {
                    s.parse::<f64>()
                        .expect("Failed to parse matrix entry as float")
                })
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
        CP {
            u: matrices[0].clone(),
            v: matrices[1].clone(),
            w: matrices[2].clone(),
        }
    }

    /// Returns a reference to the statically loaded Strassen CP decomposition matrices [U, V, W].
    pub fn get_strassen() -> &'static Self {
        STRASSEN_CP.get_or_init(Self::load_strassen)
    }
}
