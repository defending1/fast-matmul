use std::path::PathBuf;
use std::process::Command;

fn main() {
    // ----------------------------------------------------
    // C Intel MKL compilation & linking
    // ----------------------------------------------------
    let (mkl_include, mkl_lib) = if let Ok(mklroot) = std::env::var("MKLROOT") {
        let mkl_prefix = PathBuf::from(mklroot);
        let include = if mkl_prefix.join("include").exists() {
            mkl_prefix.join("include")
        } else {
            mkl_prefix.join("mkl/latest/include")
        };
        let lib = if mkl_prefix.join("lib/intel64").exists() {
            mkl_prefix.join("lib/intel64")
        } else if mkl_prefix.join("lib").exists() {
            mkl_prefix.join("lib")
        } else {
            mkl_prefix.join("mkl/latest/lib")
        };
        (include, lib)
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
        let mkl_prefix = PathBuf::from(String::from_utf8_lossy(&mkl_output.stdout).trim());
        (
            mkl_prefix.join("mkl/latest/include"),
            mkl_prefix.join("mkl/latest/lib"),
        )
    };

    println!("cargo::rustc-link-search=native={}", mkl_lib.display());
    // Force the linker to keep all MKL libraries even if not directly referenced by our object files
    // We pass them as a single comma-separated linker argument to bypass rustc reordering
    println!(
        "cargo::rustc-link-arg=-Wl,--no-as-needed,-lmkl_intel_lp64,-lmkl_gnu_thread,-lmkl_core,-lgomp,--as-needed"
    );
    println!("cargo::rustc-link-lib=dylib=pthread");
    println!("cargo::rustc-link-lib=dylib=m");
    println!("cargo::rustc-link-lib=dylib=dl");
    // Inject runtime library path (RPATH) so cargo bench executes without LD_LIBRARY_PATH
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}", mkl_lib.display());

    cc::Build::new()
        .file("ffi/matmul/mkl.c")
        .include(mkl_include)
        .compiler("clang")
        .flag("-O3")
        .flag("-march=native")
        .flag("-mtune=native")
        .compile("mkl_wrapper");

    // Recompilation triggers
    println!("cargo::rerun-if-changed=ffi/matmul/mkl.c");
    println!("cargo::rerun-if-changed=build.rs");
}
