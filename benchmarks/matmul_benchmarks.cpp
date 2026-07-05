/**
   Copyright (c) 2014-2015, Sandia Corporation
   All rights reserved.

   This file is part of fast-matmul and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause.
*/

#include "all_algorithms.hpp"
#include "common.hpp"
#ifdef __INTEL_MKL__
#include "mkl.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// Returns available system memory in bytes via /proc/meminfo on Linux.
// Returns 0 if reading fails.
unsigned long long GetAvailableMemory() {
  std::ifstream infile("/proc/meminfo");
  if (!infile.is_open()) {
    return 0;
  }
  std::string line;
  while (std::getline(infile, line)) {
    if (line.compare(0, 13, "MemAvailable:") == 0) {
      std::stringstream ss(line);
      std::string label;
      unsigned long long val;
      std::string unit;
      ss >> label >> val >> unit;
      if (unit == "kB") {
        return val * 1024;
      }
      return val;
    }
  }
  return 0;
}

// Checks if the matrix multiplication fits in memory.
bool CheckSizeSupported(int m, int k, int n) {
  // Avoid overflow using double
  double elements = (double)m * k + (double)k * n + (double)m * n;
  double bytes_per_matrix = elements * sizeof(double);

  // Safe multiplier of 10.0 to account for recursive Strassen submatrix
  // allocations
  double multiplier = 10.0;
  double estimated_bytes = bytes_per_matrix * multiplier;

  unsigned long long avail_bytes = GetAvailableMemory();
  if (avail_bytes > 0) {
    // Maintain a safety buffer (10% of available memory or at least 256MB)
    unsigned long long safety_buffer = (avail_bytes / 10 > 256 * 1024 * 1024)
                                           ? avail_bytes / 10
                                           : 256 * 1024 * 1024;
    if (estimated_bytes + safety_buffer > avail_bytes) {
      return false;
    }
  } else {
    // Fallback: restrict to 2GB if memory capacity cannot be checked via
    // /proc/meminfo
    if (bytes_per_matrix > 2.0 * 1024.0 * 1024.0 * 1024.0) {
      return false;
    }
  }
  return true;
}

// Run a single benchmark for multiplying m x k x n with num_steps of recursion.
// To just call GEMM, set num_steps to zero.
// The median of five trials is printed to the given stream.
// If run_check is true, then it also
void SingleBenchmark(std::ostream &os, int m, int k, int n, int num_steps,
                     int algorithm) {
  // Run a set number of trials and pick the median time.
  int num_trials = 2;
  std::vector<double> times(num_trials);
  for (int trial = 0; trial < num_trials; ++trial) {
    Matrix<double> A = RandomMatrix<double>(m, k);
    Matrix<double> B = RandomMatrix<double>(k, n);
    Matrix<double> C1(m, n);
    times[trial] = RunAlgorithm(algorithm, A, B, C1, num_steps);
  }

  // Spit out the median time
  std::sort(times.begin(), times.end());
  size_t ind = num_trials / 2;
  os << " " << m << " " << k << " " << n << " " << num_steps << " "
     << times[ind] << " "
     << "; " << std::flush;
}

// Runs a set of benchmarks.
void BenchmarkSet(std::ostream &os, std::vector<int> &m_vals,
                  std::vector<int> &k_vals, std::vector<int> &n_vals,
                  std::vector<int> &num_steps, int algorithm) {

  assert(m_vals.size() == k_vals.size() && k_vals.size() == n_vals.size());

  for (int curr_num_steps : num_steps) {
    os << Alg2Str(algorithm) << "_" << curr_num_steps << " = [";
    for (int i = 0; i < m_vals.size(); ++i) {
      if (!CheckSizeSupported(m_vals[i], k_vals[i], n_vals[i])) {
        std::cout << "Skipping benchmark size " << m_vals[i] << "x" << k_vals[i]
                  << "x" << n_vals[i]
                  << " as it exceeds available system memory." << std::endl;
        continue;
      }
      SingleBenchmark(os, m_vals[i], k_vals[i], n_vals[i], curr_num_steps,
                      algorithm);
    }
    os << "];" << std::endl;
  }
  os << std::endl << std::endl;
}

void SquareTest(std::ostream &os, bool full) {
  std::vector<int> m_vals;
  int limit = full ? 1048576 : 2048;
  for (int i = 2; i <= limit; i *= 2) {
    m_vals.push_back(i);
  }
  std::vector<int> num_levels = {0};
  BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, MKL);
  num_levels = {1, 2, 3};
  BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, STRASSEN);
  return;
}

void SquareTestPar(std::ostream &os, bool full) {
  std::vector<int> m_vals;
  int limit = full ? 1048576 : 2048;
  for (int i = 2; i <= limit; i *= 2) {
    m_vals.push_back(i);
  }

  std::vector<int> num_levels = {0};
#if defined(_PARALLEL_) && (_PARALLEL_ == _DFS_PAR_)
  BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, MKL);
#endif

  num_levels = {1, 2, 3};
  BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, STRASSEN);
}

void OuterTestPar(std::ostream &os) {
  std::vector<int> m_vals;
  for (int i = 3000; i <= 18000; i += 500) {
    m_vals.push_back(i);
  }
  std::vector<int> k_vals(m_vals.size(), 2800);

  std::vector<int> num_levels = {0};
#if defined(_PARALLEL_) && (_PARALLEL_ == _DFS_PAR_)
  BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels, MKL);
#endif
  num_levels = {1, 2};
  BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels, FAST424_26_257);
}

void TSSquareTestPar(std::ostream &os) {
  std::vector<int> m_vals;
  for (int i = 3000; i <= 20000; i += 500) {
    m_vals.push_back(i);
  }
  std::vector<int> k_vals(m_vals.size(), 3000);
  std::vector<int> num_levels = {0};

#if defined(_PARALLEL_) && (_PARALLEL_ == _DFS_PAR_)
  BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels, MKL);
#endif
  num_levels = {1, 2};
  BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels, FAST433_29_234);
}

void SquareBenchmark(std::ostream &os, int which) {
  std::vector<int> m_vals;
  for (int i = 2; i <= 8192; i *= 2) {
    m_vals.push_back(i);
  }
  std::vector<int> num_levels_MKL = {0};
  std::vector<int> num_levels = {1, 2};

  switch (which) {
  case 0:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels_MKL, MKL);
    break;
  case 1:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, BINI322_10_52_APPROX);
    break;
  case 2:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, STRASSEN);
    break;
  case 3:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST322_11_50);
    break;
  case 4:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST422_14_84);
    break;
  case 5:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST323_15_103);
    break;
  case 6:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST332_15_103);
    break;
  case 7:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST522_18_99);
    break;
  case 8:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST252_18_99);
    break;
  case 9:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST234_20_144);
    break;
  case 10:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST324_20_144);
    break;
  case 11:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST423_20_144);
    break;
  case 12:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST342_20_144);
    break;
  case 13:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST333_23_152);
    break;
  case 14:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST424_26_257);
    break;
  case 15:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST442_26_257);
    break;
  case 16:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST433_29_234);
    break;
  case 17:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, FAST343_29_234);
    break;
  case 18:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, SMIRNOV336_40_960);
    break;
  case 19:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, SMIRNOV363_40_960);
    break;
  case 20:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, SMIRNOV633_40_960);
    break;
  case 21:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels,
                 SCHONHAGE333_21_117_APPROX);
    break;
  case 22:
    BenchmarkSet(os, m_vals, m_vals, m_vals, num_levels, CLASSICAL423);
    break;
  default:
    throw std::logic_error("Unknown algorithm");
  }

  return;
}

// (N, k, N) for fixed k ~ 2000
void OuterProductBenchmark(std::ostream &os, int which) {
  std::vector<int> m_vals;
#ifdef _PARALLEL_
  for (int i = 3000; i <= 18000; i += 500) {
    m_vals.push_back(i);
  }
  std::vector<int> k_vals(m_vals.size(), 2800);
#else
  for (int i = 2000; i <= 12000; i += 500) {
    m_vals.push_back(i);
  }
  std::vector<int> k_vals(m_vals.size(), 1600);
#endif

  std::vector<int> num_levels_MKL = {0};
  std::vector<int> num_levels = {1, 2};

  switch (which) {
  case 0:
    BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels_MKL, MKL);
    break;
  case 1:
    BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels, FAST424_26_257);
    break;
  case 2:
    BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels, FAST433_29_234);
    break;
  case 3:
    BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels, FAST323_15_103);
    break;
  case 4:
    BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels, FAST522_18_99);
    break;
  case 5:
    BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels, FAST423_20_144);
    break;
  case 6:
    BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels, STRASSEN);
    break;
  case 7:
    BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels, BINI322_10_52_APPROX);
    break;
  case 8:
    BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels,
                 SCHONHAGE333_21_117_APPROX);
    break;
  case 9:
    BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels, SMIRNOV633_40_960);
    break;
  case 10:
    BenchmarkSet(os, m_vals, k_vals, m_vals, num_levels, CLASSICAL423);
    break;
  default:
    throw std::logic_error("Unknown algorithm");
  }
}

// (N, k, k) for fixed k ~ 2000
void TSSquareBenchmark(std::ostream &os, int which) {
  std::vector<int> m_vals;
#ifdef _PARALLEL_
  for (int i = 3000; i <= 20000; i += 500) {
    m_vals.push_back(i);
  }
  std::vector<int> k_vals(m_vals.size(), 3000);
#else
  for (int i = 10000; i <= 18000; i += 500) {
    m_vals.push_back(i);
  }
  std::vector<int> k_vals(m_vals.size(), 2400);
#endif

  std::vector<int> num_levels_MKL = {0};
  std::vector<int> num_levels = {1, 2};
  switch (which) {
  case 0:
    BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels_MKL, MKL);
    break;
  case 1:
    BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels, FAST424_26_257);
    break;
  case 2:
    BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels, FAST433_29_234);
    break;
  case 3:
    BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels, FAST323_15_103);
    break;
  case 4:
    BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels, FAST522_18_99);
    break;
  case 5:
    BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels, FAST423_20_144);
    break;
  case 6:
    BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels, STRASSEN);
    break;
  case 7:
    BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels, BINI322_10_52_APPROX);
    break;
  case 8:
    BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels,
                 SCHONHAGE333_21_117_APPROX);
    break;
  case 9:
    BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels, SMIRNOV633_40_960);
    break;
  case 10:
    BenchmarkSet(os, m_vals, k_vals, k_vals, num_levels, CLASSICAL423);
    break;
  }
}

int main(int argc, char **argv) {
  // Re-route execution to a job-specific copy if running in a cluster job environment.
  const char* slurm_job_id = std::getenv("SLURM_JOB_ID");
  const char* pbs_job_id = std::getenv("PBS_JOBID");
  const char* env_run_id = std::getenv("RUN_ID");
  std::string job_id = "";
  if (slurm_job_id && *slurm_job_id) {
    job_id = slurm_job_id;
  } else if (pbs_job_id && *pbs_job_id) {
    job_id = pbs_job_id;
  } else if (env_run_id && *env_run_id) {
    job_id = env_run_id;
  }

  if (!job_id.empty()) {
    char path_buf[1024];
    ssize_t len = readlink("/proc/self/exe", path_buf, sizeof(path_buf) - 1);
    if (len != -1) {
      path_buf[len] = '\0';
      std::string current_exe(path_buf);
      std::string suffix = "_" + job_id;
      if (current_exe.length() < suffix.length() || 
          current_exe.compare(current_exe.length() - suffix.length(), suffix.length(), suffix) != 0) {
        
        std::string unique_exe = current_exe + suffix;
        
        // Copy the executable file
        std::ifstream src(current_exe, std::ios::binary);
        std::ofstream dst(unique_exe, std::ios::binary);
        if (src && dst) {
          dst << src.rdbuf();
          src.close();
          dst.close();
          
          chmod(unique_exe.c_str(), 0755);
          execv(unique_exe.c_str(), argv);
          std::cerr << "Warning: Failed to exec job-dependent C++ binary, falling back to original." << std::endl;
        }
      }
    }
  }

  // Parse command-line -full flag first, stripping it to avoid options crash
  bool full = false;
  std::vector<char *> new_argv;
  new_argv.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "-full") {
      full = true;
    } else {
      new_argv.push_back(argv[i]);
    }
  }
  int new_argc = new_argv.size();
  auto opts = GetOpts(new_argc, new_argv.data());

  // Create the generated output directory if it doesn't exist
  mkdir("benchmarks/generated", 0755);

  std::string out_filename = "benchmarks/generated/benchmarks";
  if (!job_id.empty()) {
    out_filename += "_" + job_id;
  }
  out_filename += ".txt";

  std::ofstream fout(out_filename);
  if (!fout.is_open()) {
    std::cerr << "Error: could not open " << out_filename << std::endl;
    return 1;
  }

  // Run all <N, N, N> benchmarks
  if (OptExists(opts, "square_all")) {
    for (int i = 0; i <= 21; ++i) {
      SquareBenchmark(fout, i);
      fout.flush();
    }
  }

  // Run a single <N, N, N> benchmark
  if (OptExists(opts, "square")) {
    int which = GetIntOpt(opts, "square");
    SquareBenchmark(fout, which);
    fout.flush();
  }

  // Run <N, k, N> benchmark for fixed k
  if (OptExists(opts, "outer_prod_like")) {
    int which = GetIntOpt(opts, "outer_prod_like");
    OuterProductBenchmark(fout, which);
    fout.flush();
  }

  // Run <N, k, k> benchmark for fixed k
  if (OptExists(opts, "ts_square_like")) {
    int which = GetIntOpt(opts, "ts_square_like");
    TSSquareBenchmark(fout, which);
    fout.flush();
  }

  // Functions for testing
  if (OptExists(opts, "square_test")) {
    SquareTest(fout, full);
    fout.flush();
  }
  if (OptExists(opts, "square_test_par")) {
    SquareTestPar(fout, full);
    fout.flush();
  }
  if (OptExists(opts, "outer_test_par")) {
    OuterTestPar(fout);
    fout.flush();
  }
  if (OptExists(opts, "ts_square_test_par")) {
    TSSquareTestPar(fout);
    fout.flush();
  }

  // Clean up job-dependent copy on exit if running the clone
  if (!job_id.empty()) {
    char path_buf[1024];
    ssize_t len = readlink("/proc/self/exe", path_buf, sizeof(path_buf) - 1);
    if (len != -1) {
      path_buf[len] = '\0';
      std::string current_exe(path_buf);
      std::string suffix = "_" + job_id;
      if (current_exe.length() >= suffix.length() && 
          current_exe.compare(current_exe.length() - suffix.length(), suffix.length(), suffix) == 0) {
        unlink(current_exe.c_str());
      }
    }
  }

  return 0;
}
