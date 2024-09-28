Here is a **README** that includes details about both the matrix multiplication programs you provided:

---

# Matrix Multiplication Optimizations

This repository contains two matrix multiplication programs, each demonstrating different approaches to optimize performance through various techniques mentioned in Anatomy of High-Performance Matrix Multiplication by KAZUSHIGE GOTO and ROBERT A. VAN DE GEIJN.

---

## Programs:

### 1. **Matrix Multiplication with GEBP Optimization (`goto_van.c`)**

This program performs matrix multiplication using the ***GEBP** algorithm mentioned in Anatomy of High-Performance Matrix. It optimizes cache usage by breaking the matrices into blocks that fit within the L1 and L2 caches of modern CPUs. All matrices are accessed and stored in **column-major order** via the terms of the project.

#### Features:
- **GEBP Algorithm**: Optimizes matrix multiplication by processing smaller sub-blocks of matrices.
- **Configurable Block Sizes**: You can adjust block sizes to fit different cache sizes and registers.
- **Column-Major Storage**: Matrices are stored in column-major order to align with certain linear algebra libraries and hardware optimizations.
- **Performance Output**: The program outputs detailed performance metrics, including GFLOPS and cache usage, and saves them to a CSV file.

#### Compilation:

```bash
gcc -O3 -march=native goto_van.c matrix_ops.c -o gotovan -lm
```

#### Usage:

Run the program normally:

```bash
./gotovan
```

Or in debug mode (which prints intermediate matrices):

```bash
./gotovan 1
```

#### Output:

The program generates a CSV file located at `../output/gotovan.csv` with the following columns:
- Block sizes (`k_c`, `n_r`, `m_r`)
- GFLOPS
- Time taken (in seconds)
- Cache usage (L1 and L2)

#### Example Output:

```bash
Matrix Sizes: 4096x4096
Block Sizes: m_c = 170, k_c = 170, n_r = 8, m_r = 16
Time taken: 12.345678 seconds
GFLOPS: 200.123456
GFLOPS Utilization: 0.95
B block size (KB)/L1 Cache size: 8/32.000000
A block size (KB)/L2 Cache size: 256/256.000000
```

---

### 2. **Matrix Multiplication with Different Loop Orderings (`matmulp.c`)**

This program demonstrates matrix multiplication using six different loop orderings. Each of the following loop permutations is timed and measured for performance:

- `ijk` (Standard)
- `ikj`
- `kij`
- `kji`
- `jki`
- `jik`

The program measures the performance of each permutation in terms of GFLOPS and outputs the results to the console and a CSV file.

#### Features:
- **Six Loop Orderings**: Tests different loop orderings for matrix multiplication to determine the best-performing one.
- **Performance Metrics**: Records the time taken and GFLOPS for each permutation and outputs the data to a CSV file.
- **Matrix Reset**: Resets the result matrix before each multiplication to ensure correct measurements.

#### Compilation:

```bash
gcc -fopenmp -O3 -march=native matmulp.c -o matmulp -lm
```

#### Usage:

To run the program:

```bash
./matmulp
```

#### Output:

The program generates a CSV file located at `../output/matmulp_results.csv` with the following columns:
- Loop order (`ijk`, `ikj`, etc.)
- Time taken (in seconds)
- GFLOPS

#### Example Output:

```bash
Matrix Multiplications:
ijk order: 2.345678 seconds. GFLOPS: 150.123456
ikj order: 2.123456 seconds. GFLOPS: 160.123456
kij order: 2.567890 seconds. GFLOPS: 140.567890
...
```

Each of these orders will exhibit different performance characteristics based on cache locality and memory access patterns.

---

## Program Files:

### `goto_van.c`:
- Matrix multiplication using the **Generalized Blocked Panel (GEBP)** algorithm.
- Outputs performance metrics including GFLOPS and cache usage.

### `matmulp.c`:
- Matrix multiplication using six different loop orderings.
- Outputs performance metrics and writes results to a CSV file.

---

## Contributors:
- Aman Hogan-Bailey
- Anatomy of High-Performance Matrix Multiplication by KAZUSHIGE GOTO and ROBERT A. VAN DE GEIJN
- University of Texas at Arlington