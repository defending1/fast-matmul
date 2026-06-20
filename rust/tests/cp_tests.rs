use fast_matmul::cp::CP;
use std::fs;

#[test]
fn test_load_all_exact_algorithms() {
    let paths = [
        "codegen/algorithms",
        "../codegen/algorithms",
        "../../codegen/algorithms",
    ];
    let mut base_dir = None;
    for base in &paths {
        if fs::metadata(base).is_ok() {
            base_dir = Some(base);
            break;
        }
    }
    let base_dir = base_dir.expect("Could not find codegen/algorithms directory");
    let entries = fs::read_dir(base_dir).unwrap();
    for entry in entries {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            let name = path.file_name().unwrap().to_str().unwrap();
            if name.contains("approx") || name.starts_with('.') {
                continue;
            }
            // Skip subexpression eliminated algorithms
            if fs::read_to_string(&path)
                .map(|c| c.contains("Substitution information"))
                .unwrap_or(false)
            {
                continue;
            }
            println!("Testing loading: {}", name);
            let cp = CP::load(name);
            assert!(cp.m > 0);
            assert!(cp.n > 0);
            assert!(cp.p > 0);
            assert!(cp.rank > 0);
        }
    }
}
