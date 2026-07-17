#set page(
  paper: "presentation-16-9",
  fill: rgb("#1a365d"),
  margin: 0cm,
)
#set text(
  font: "Liberation Sans",
  size: 20pt,
  fill: white,
)

// Math shortcuts from tensors.sty mapping
#let topp = math.compose
#let krn = math.times.o
#let krp = math.dot.o
#let had = math.ast
#let dsqr(..args) = math.bracket.l.stroked + args.pos().join(",") + math.bracket.r.stroked
#let angb(..args) = math.chevron.l + args.pos().join(",") + math.chevron.r
#let Vec = math.op("vec")

// Title slide layout
#align(center + horizon)[
  #block(
    width: 85%,
    inset: 2.2em,
    radius: 12pt,
    fill: rgb("#112446").lighten(10%),
    stroke: 1.5pt + rgb("#3182ce"),
  )[
    #text(size: 26pt, weight: "bold", fill: rgb("#ebf8ff"))[Evaluating Fast Matrix Multiplication Algorithms In Rust]
    #v(0.4em)
    #text(size: 18pt, style: "italic", fill: rgb("#90cdf4"))[Computational Laboratory Project Presentation]
    #v(1.8em)
    #grid(
      columns: 2,
      gutter: 1fr,
      align: (left, right),
      [
        #text(size: 13pt, fill: rgb("#a0aec0"))[Author:] \
        #text(size: 15pt, weight: "bold")[Alberto Defendi]
      ],
      [
        #text(size: 13pt, fill: rgb("#a0aec0"))[Supervisor:] \
        #text(size: 15pt, weight: "bold")[Leonardo Robol]
      ]
    )
  ]
]

// Reset page setting for main slides
#set page(
  fill: rgb("#f7fafc"),
  margin: (x: 1.5cm, top: 1.2cm, bottom: 1.0cm),
  header: context {
    let page = counter(page).get().first()
    if page > 1 {
      grid(
        columns: (1fr, auto),
        align: (left, right),
        text(size: 10pt, fill: rgb("#718096"))[Evaluating Fast Matrix Multiplication in Rust],
        text(size: 10pt, fill: rgb("#718096"))[Section #context counter(heading).display()]
      )
      v(-0.4em)
      line(length: 100%, stroke: 0.5pt + rgb("#cbd5e0"))
    }
  },
  footer: context {
    let page = counter(page).get().first()
    if page > 1 {
      grid(
        columns: (1fr, auto),
        align: (left, right),
        text(size: 10pt, fill: rgb("#a0aec0"))[Università di Pisa],
        text(size: 10pt, fill: rgb("#718096"), weight: "bold")[#page]
      )
    }
  }
)
#set text(fill: rgb("#2d3748"), size: 14pt) // Slightly smaller body size to prevent overflow

// Template helpers
#let slide(title: none, body) = {
  pagebreak(weak: true)
  if title != none {
    block(width: 100%, inset: (bottom: 0.4em))[
      #text(fill: rgb("#1a365d"), size: 20pt, weight: "bold")[#title]
      #v(-0.2em)
      #line(length: 100%, stroke: 1.5pt + rgb("#3182ce"))
    ]
  }
  body
}

#let definition(title, body) = block(
  fill: rgb("#ebf8ff"),
  stroke: (left: 4pt + rgb("#3182ce")),
  inset: 0.6em,
  radius: (right: 4pt),
  width: 100%,
  above: 0.4em,
  below: 0.4em,
)[
  #text(weight: "bold", fill: rgb("#2b6cb0"), size: 13pt)[Definition: #title] \
  #body
]

#let theorem(title, body) = block(
  fill: rgb("#f0fff4"),
  stroke: (left: 4pt + rgb("#38a169")),
  inset: 0.6em,
  radius: (right: 4pt),
  width: 100%,
  above: 0.4em,
  below: 0.4em,
)[
  #text(weight: "bold", fill: rgb("#276749"), size: 13pt)[Theorem: #title] \
  #body
]

#let highlight-box(body) = block(
  fill: rgb("#fffaf0"),
  stroke: 1pt + rgb("#dd6b20"),
  inset: 0.6em,
  radius: 4pt,
  width: 100%,
  above: 0.4em,
  below: 0.4em,
)[#body]

#slide(title: [Introduction])[
- *Matrix Multiplication:* One of the most fundamental computations in numerical linear algebra and scientific computing.
- *Fast Algorithms:* Perform asymptotically fewer floating-point operations and require less data communication than the classical $O(N^3)$ algorithm.
- *Project Goals:*
- Implement fast matrix multiplication using arbitrary CP tensor decompositions in Rust.
- Evaluate native Rust linear algebra library (`faer`) against optimized vendor GEMMs (Intel `MKL dgemm` via FFI).
- Handle general dimensions via memory-efficient *dynamic peeling* instead of zero-padding.
- Benchmark sequential and parallel execution using multiple scheduling strategies.
]

#slide(title: [Tensors & Outer Products])[
  #definition([3-Way Tensor], [
    A 3-dimensional (or 3-way) tensor is a multidimensional array $T in RR^(M times N times P)$. Its slices obtained by fixing one index are matrices. For example, the $k$-th frontal slice of $T$ is the matrix $X_k$.
  ])

  #definition([Outer Product], [
    For vectors $u in RR^M, v in RR^N, w in RR^P$, the rank-1 outer product tensor $T = u topp v topp w in RR^(M times N times P)$ has entries:
    $ x_(i j k) = u_i v_j w_k $
    The rank of a tensor $T$ is the minimum number of rank-1 tensors that generate $T$ as their sum.
  ])
]

#slide(title: [Low-Rank Tensor Decompositions])[
  #definition([CP Decomposition], [
    The CANDECOMP/PARAFAC (CP) decomposition of rank $R$ of a tensor $T$ is:
    $ T = dsqr(U, V, W) = sum_(r=1)^R u_r topp v_r topp w_r $
    where $U in RR^(M times R)$, $V in RR^(N times R)$, and $W in RR^(P times R)$ are factor matrices with columns $u_r$, $v_r$, and $w_r$.
  ])

  #definition([Mode-k Product], [
    The mode-$k$ product between a tensor $T in RR^(N_1 times N_2 times N_3)$ and a matrix $U in RR^(R times N_k)$ for $k in {1,2,3}$ is denoted by $Y = T times_k U$. Its matricized representation is:
    $ Y_((k)) = U T_((k)) $
  ])
]

#slide(title: [Bilinear Forms via Low-Rank Decompositions])[
- A bilinear form $f : X times Y -> Z$ is represented by a tensor $T$:
$ f = T times_1 x^T times_2 y^T $
- Using a CP decomposition $T = dsqr(U, V, W)$ of rank $R$:
$ f(x, y) = sum_(l=1)^R (u_l^T x) dot (v_l^T y) w_l $
- Evaluating $f(x, y)$ requires exactly $R$ active scalar multiplications and $O(R)$ additions.
- For matrix multiplication $C = A B$, representing the form $angb(M, N, P)$, we have:
$ Vec(C^T) = T times_1 Vec(A)^T times_2 Vec(B)^T $
- Given a CP decomposition of $T = dsqr(U, V, W)$, we get:
$ Vec(C^T) = sum_(l=1)^R (s_l dot t_l) w_l $
where $s_l = u_l^T Vec(A)$ and $t_l = v_l^T Vec(B)$.
]

#slide(title: [Strassen's Algorithm & CP Decomposition])[
- *Strassen's Algorithm* for $angb(2, 2, 2)$ uses $R=7$ multiplications and $18$ additions:
#align(center)[
  #text(size: 11pt)[
    $ U = mat(1, 0, 1, 0, 1, -1, 0; 0, 0, 0, 0, 1, 0, 1; 0, 1, 0, 0, 0, 1, 0; 1, 1, 0, 1, 0, 0, -1), \
    V = mat(1, 1, 0, -1, 0, 1, 0; 0, 0, 1, 0, 0, 1, 0; 0, 0, 0, 1, 0, 0, 1; 1, 0, -1, 0, 1, 0, 1), \
    W = mat(1, 0, 0, 1, -1, 0, 1; 0, 1, 0, 1, 0, 0, 0; 0, 0, 1, 0, 1, 0, 0; 1, -1, 1, 1, 0, 0, 0) $
  ]
]
- This reduces arithmetic complexity from $O(N^3)$ to $O(N^(log_2 7)) = O(N^2.81)$.
- Fast algorithms can be constructed for general base cases $angb(m, n, p)$ using CP decompositions of rank $R$.
]

#slide(title: [Summary of Fast Matrix Algorithms])[
  #align(center)[
    #table(
      columns: (auto, auto, auto, auto),
      align: (left, center, center, center),
      stroke: 0.5pt + rgb("#cbd5e0"),
      fill: (col, row) => if row == 0 { rgb("#ebf8ff") } else { none },
      [*Algorithm Base Case*], [*Multiplications ($R$)*], [*Classical ($m n p$)*], [*Speedup per Step*],
      [$angb(2, 2, 2)$ (Strassen)], [7], [8], [14%],
      [$angb(2, 2, 3)$], [11], [12], [9%],
      [$angb(2, 2, 4)$], [14], [16], [14%],
      [$angb(2, 2, 5)$], [18], [20], [11%],
      [$angb(2, 3, 3)$], [15], [18], [20%],
      [$angb(2, 3, 4)$], [20], [24], [20%],
      [$angb(2, 4, 4)$], [26], [32], [23%],
      [$angb(3, 3, 3)$], [23], [26], [17%],
      [$angb(3, 3, 4)$], [29], [36], [24%],
      [$angb(3, 3, 6)$], [40], [54], [35%],
    )
  ]
]

#slide(title: [Dynamic Peeling])[
- Traditional fast algorithms require dimensions to match the base case power.
- *Dynamic Peeling:* A memory-efficient alternative to zero-padding.
- Split $A$ and $B$ into core components and peeled boundary vectors/scalars:
$ A = mat(A_11, a_12; a_21, a_22), \qquad B = mat(B_11, b_12; b_21, b_22) $
- Recombine using a combination of the core product and boundary corrections:
$ C = mat(C_11, c_12; c_21, c_22) = mat(A_11 B_11 + a_12 b_21, A_11 b_12 + a_12 b_22; a_21 B_11 + a_22 b_21, a_21 b_12 + a_22 b_22) $
- Leaf-node $A_11 B_11$ is solved recursively, while all boundary corrections are consolidated into single, large GEMM calls to maximize cache locality and vendor BLAS efficiency.
]

#slide(title: [Shared Memory Parallel Scheduling])[
  To balance computational load and avoid recursion tree overhead, we benchmarked three scheduling strategies in Rust using the Rayon library:

- *DFS (Depth-First-Search):*
- Evaluates the recursion tree sequentially.
- Uses all available threads inside vendor leaf GEMM calls.
- High thread under-utilization during split and addition phases.
- *BFS (Breadth-First-Search):*
- Parallelizes all independent recursive subproblems at the top levels.
- Rayon executes recursive branches in parallel.
- *Hybrid:*
- Compensates for BFS load imbalance.
- BFS task parallelism is applied to the first $R^L - (R^L mod P)$ tasks (where $P$ is the thread count).
- DFS (all threads on single GEMM) is used on the remaining $R^L mod P$ tasks.
]

#slide(title: [Execution Platform & Setup])[
- *Hardware:* One node of a dual-socket AMD EPYC 7763 server (64 physical cores, 2 TB RAM).
- *Software environment:*
- GCC 13.2.0, Intel OneAPI compilers 2025.0.4, Intel OneAPI MKL.
- Rust Nightly (2026-06-23) utilizing LLVM 22 for optimized AVX2 and FMA generation.
- *Linear Algebra Backends:*
- *Intel MKL dgemm:* Vendor library called via custom FFI wrapper.
- *faer matmul:* Native Rust high-performance linear algebra library.
- *Metric:*
$ text("Effective GFLOPS") = frac(2 M N P - M P, text("time in seconds") dot 10^9) $
which normalizes performance based on the classical floating-point operation count.
]

#slide(title: [Performance of Base GEMM Libraries])[
  #align(center + horizon)[
    #image("figures/mkl_faer_comparison.pdf", height: 85%)
  ]
]

#slide(title: [Sequential Crossover Point])[
  #align(center + horizon)[
    #image("figures/sequential_grid_plot.pdf", height: 85%)
  ]
]

#slide(title: [Parallel Crossover (Normalized per Core)])[
  #align(center + horizon)[
    #image("figures/parallel_grid_plot_strassen_only.pdf", height: 85%)
  ]
]

#slide(title: [Parallel Cutoffs: MKL vs. Faer Bases])[
  #align(center + horizon)[
    #grid(
      columns: (1fr, 1fr),
      gutter: 1.5em,
      image("figures/parallel_grid_plot_dgemm.pdf", width: 100%),
      image("figures/parallel_grid_plot_faer.pdf", width: 100%),
    )
  ]
]

#slide(title: [Finding the Optimal Size Cutoff])[
  #align(center + horizon)[
    #grid(
      columns: (1fr, 1fr),
      gutter: 1.5em,
      image("figures/parallel_cutoff_grid_plot_dgemm.pdf", width: 100%),
      image("figures/parallel_cutoff_grid_plot_faer.pdf", width: 100%),
    )
  ]
]

#slide(title: [Comparison with Benson & Ballard C++ Reference])[
  #align(center + horizon)[
    #image("figures/compare_ballard.pdf", height: 85%)
  ]
]

#slide(title: [Conclusions & Future Work])[
  #highlight-box([
    *Key Takeaways:*
- Fast matrix multiplication algorithms successfully outperform optimized vendor/native GEMM libraries in both sequential and parallel execution.
- Crossover points are highly practical ($N=8192$ sequential, $N=16384$ parallel).
- Rayon's work-stealing scheduler provides superior load-balancing and scaling compared to OpenMP at deeper levels of recursion.
  ])

- *Future Directions:*
- *Profiling & SIMD:* Finer refactoring of matrix split/addition routines using SIMD vectorization.
- *Other GEMM Backends:* Add support for OpenBLAS, BLIS, or other open-source libraries.
- *Distributed/GPU version:* Port to GPU (CUDA/WebGPU) or distributed clusters to scale beyond $N=65536$.
- *Architectural Comparisons:* Evaluate on high-core-count Intel Xeon processors.
]
