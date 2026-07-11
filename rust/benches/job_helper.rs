use std::os::unix::process::CommandExt;

/// Re-routes execution to a job-specific copy if running in a cluster job environment.
pub fn handle_job_dependent_execution() {
    let job_id = std::env::var("SLURM_JOB_ID")
        .or_else(|_| std::env::var("PBS_JOBID"))
        .or_else(|_| std::env::var("RUN_ID"))
        .ok();

    if let (Some(job_id), Ok(current_exe)) = (job_id, std::env::current_exe()) {
            let suffix = format!("_{}", job_id);
            let current_exe_str = current_exe.to_string_lossy().into_owned();
            if !current_exe_str.ends_with(&suffix) {
                let unique_exe_str = format!("{}{}", current_exe_str, suffix);
                let unique_exe = std::path::PathBuf::from(&unique_exe_str);

                if std::fs::copy(&current_exe, &unique_exe).is_ok() {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        if let Ok(metadata) = std::fs::metadata(&unique_exe) {
                            let mut perms = metadata.permissions();
                            perms.set_mode(0o755);
                            let _ = std::fs::set_permissions(&unique_exe, perms);
                        }
                    }

                    let args: Vec<String> = std::env::args().collect();
                    let err = std::process::Command::new(&unique_exe)
                        .args(&args[1..])
                        .exec();
                    eprintln!("Warning: Failed to exec job-dependent Rust binary: {:?}", err);
                }
            }
        }
    }

/// Clean up job-dependent clone on exit if we are running the clone
pub fn cleanup_job_dependent_execution() {
    if let Ok(current_exe) = std::env::current_exe() {
        let current_exe_str = current_exe.to_string_lossy().into_owned();
        let job_id = std::env::var("SLURM_JOB_ID")
            .or_else(|_| std::env::var("PBS_JOBID"))
            .or_else(|_| std::env::var("RUN_ID"));
        if let Ok(job_id) = job_id {
            let suffix = format!("_{}", job_id);
            if current_exe_str.ends_with(&suffix) {
                let _ = std::fs::remove_file(current_exe);
            }
        }
    }
}
