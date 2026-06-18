# Matrix Vectorization Layout: PDF vs. Rust Implementation

In the matrix multiplication implementation, there is a layout discrepancy between the theoretical formulation in the PDF and the actual Rust codebase in [matmul.rs](file:///home/alberto/Data/pisa/fast-matmul/rust/src/matmul.rs). This document outlines how it is resolved.

## 1. Vectorization Ordering Discrepancy

At the end of page 3 of [MatrixMultiplication.pdf](file:///home/alberto/Data/pisa/fast-matmul/report/refs/MatrixMultiplication.pdf), the block vectorizations for $U$ and $V$ are defined in **column-major** order:

$$\text{vec}(U)_{PDF} = \begin{bmatrix} U_{11} \\ U_{21} \\ U_{12} \\ U_{22} \end{bmatrix}, \quad \text{vec}(V)_{PDF} = \begin{bmatrix} V_{11} \\ V_{21} \\ V_{12} \\ V_{22} \end{bmatrix}$$

However, in the Rust implementation of [matmul_cp](file:///home/alberto/Data/pisa/fast-matmul/rust/src/matmul.rs#L172-L205), the blocks are vectorized in **row-major** order:

$$\text{vec}(U)_{Rust} = \begin{bmatrix} U_{11} \\ U_{12} \\ U_{21} \\ U_{22} \end{bmatrix}, \quad \text{vec}(V)_{Rust} = \begin{bmatrix} V_{11} \\ V_{12} \\ V_{21} \\ V_{22} \end{bmatrix}$$

This swaps the middle elements (index 1 and index 2) of the vectors.

---

## 2. The Resolution: Swapped Coefficient Rows

Because the vectorization in the code swaps the middle two elements, the coefficient matrices loaded from [codegen/algorithms/strassen](file:///home/alberto/Data/pisa/fast-matmul/codegen/algorithms/strassen) also have their **second and third rows swapped** compared to the matrices printed in the PDF:

| Matrix / Row | PDF Version (Column-Major) | File Version (Row-Major) |
| :--- | :--- | :--- |
| **Matrix A (Row 2)** | `0 1 0 0 0 1 0` | `0 0 0 0 1 0 1` |
| **Matrix A (Row 3)** | `0 0 0 0 1 0 1` | `0 1 0 0 0 1 0` |
| **Matrix B (Row 2)** | `0 0 0 1 0 0 1` | `0 0 1 0 0 1 0` |
| **Matrix B (Row 3)** | `0 0 1 0 0 1 0` | `0 0 0 1 0 0 1` |

Because Row 2 and Row 3 are swapped in the coefficient file, the dot products $a_l^T \text{vec}(U)$ and $b_l^T \text{vec}(V)$ evaluate to the exact same algebraic terms in both formulations. 

For the output matrix $W$, both the PDF and Rust use row-major order (`[W11; W12; W21; W22]`), meaning Matrix $C$ does not require row swaps and is identical in both.

---

## 3. Why was it implemented this way?

The project's code generator (`codegen/`) and the algorithm files (like [codegen/algorithms/strassen](file:///home/alberto/Data/pisa/fast-matmul/codegen/algorithms/strassen)) were originally created for a C++ codebase which uses row-major ordering. 

To remain compatible with these pre-existing algorithm files without needing to rewrite/convert the files during runtime, the Rust implementation adopts the row-major layout.
