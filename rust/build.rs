use std::path::PathBuf;
use std::process::Command;

fn main() {
    // ----------------------------------------------------
    // C Intel MKL compilation & linking
    // ----------------------------------------------------
    let mkl_env_vars = ["MKLROOT", "MKL_ROOT", "INTEL_ONEAPI_MPI_ROOT", "I_MPI_ROOT"];
    let mkl_env_val = mkl_env_vars.iter().find_map(|&var| std::env::var(var).ok());

    let mkl_prefix = if let Some(mklroot) = mkl_env_val {
        PathBuf::from(mklroot)
    } else {
        let mkl_output = Command::new("spack")
            .args(["location", "-i", "intel-oneapi-mkl"])
            .output()
            .expect("Failed to execute spack command for MKL. Is spack sourced & in your PATH?");

        assert!(
            mkl_output.status.success(),
            "spack location -i intel-oneapi-mkl failed. Make sure MKL is installed via Spack.\nError: {}",
            String::from_utf8_lossy(&mkl_output.stderr)
        );
        PathBuf::from(String::from_utf8_lossy(&mkl_output.stdout).trim())
    };

    let mkl_include = if mkl_prefix.join("include").exists() {
        mkl_prefix.join("include")
    } else if mkl_prefix.join("mkl/latest/include").exists() {
        mkl_prefix.join("mkl/latest/include")
    } else {
        mkl_prefix.join("mkl").join("latest").join("include")
    };

    let mkl_lib = if mkl_prefix.join("lib/intel64").exists() {
        mkl_prefix.join("lib/intel64")
    } else if mkl_prefix.join("lib").exists() {
        mkl_prefix.join("lib")
    } else if mkl_prefix.join("mkl/latest/lib").exists() {
        mkl_prefix.join("mkl/latest/lib")
    } else {
        mkl_prefix.join("mkl").join("latest").join("lib")
    };

    println!("cargo::rustc-link-search=native={}", mkl_lib.display());

    // Locate libiomp5.so directory in the compiler runtime paths
    let spack_prefix = if mkl_prefix.join("compiler").exists() {
        mkl_prefix.clone()
    } else if mkl_prefix.parent().map_or(false, |p| p.join("compiler").exists()) {
        mkl_prefix.parent().unwrap().to_path_buf()
    } else if mkl_prefix.parent().and_then(|p| p.parent()).map_or(false, |p| p.join("compiler").exists()) {
        mkl_prefix.parent().unwrap().parent().unwrap().to_path_buf()
    } else {
        mkl_prefix.clone()
    };

    let iomp5_dir = if spack_prefix.join("compiler/latest/lib").exists() {
        Some(spack_prefix.join("compiler/latest/lib"))
    } else if spack_prefix.join("compiler/2026.0/lib").exists() {
        Some(spack_prefix.join("compiler/2026.0/lib"))
    } else {
        None
    };

    if let Some(ref dir) = iomp5_dir {
        println!("cargo::rustc-link-search=native={}", dir.display());
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }

    // Force the linker to keep all MKL libraries even if not directly referenced by our object files
    // We pass them as a single comma-separated linker argument to bypass rustc reordering
    println!(
        "cargo::rustc-link-arg=-Wl,--no-as-needed,-lmkl_intel_lp64,-lmkl_intel_thread,-lmkl_core,-liomp5,--as-needed"
    );
    println!("cargo::rustc-link-lib=dylib=pthread");
    println!("cargo::rustc-link-lib=dylib=m");
    println!("cargo::rustc-link-lib=dylib=dl");
    // Inject runtime library path (RPATH) so cargo bench executes without LD_LIBRARY_PATH
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}", mkl_lib.display());

    let target_cpu = std::env::var("TARGET_CPU").unwrap_or_else(|_| "native".to_string());

    cc::Build::new()
        .file("ffi/matmul/mkl.c")
        .include(mkl_include)
        .compiler("gcc")
        .flag("-O3")
        .flag(format!("-march={}", target_cpu))
        .flag(format!("-mtune={}", target_cpu))
        .compile("mkl_wrapper");

    // Recompilation triggers
    println!("cargo::rerun-if-changed=ffi/matmul/mkl.c");
    println!("cargo::rerun-if-changed=build.rs");
}
