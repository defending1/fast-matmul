# Benchmarks results

# 30.06

Added strassen lines from Ballard et Al.

![img](./generated/plots/benchmark_plot2906_ballard.png)

## 29.06

Remarks:

- MKL could surpass faer for bigger sizes.
- Larger fallback size to vendor library increases performance of fast
  algorithms.

### Switch to base when n<=256.

![img](./generated/plots/benchmark_plot2906_base_256.png)

### Switch to base when n <= 512.

![img](./generated/plots/benchmark_plot2906_base_512.png)

### Switch to base when n <= 1024.

![img](./generated/plots/benchmark_plot2906_base_1024.png)
