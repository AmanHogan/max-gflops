# Summary
### Matrix Multiplication Optimizations
This repository contains two matrix multiplication programs, each demonstrating different approaches to optimize performance through various techniques mentioned in [Anatomy of High-Performance Matrix Multiplication by KAZUSHIGE GOTO and ROBERT A. VAN DE GEIJN](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf). This project is apart of the 2024 Fall Semester at UTA for parallel processing.

---

# Prerequisites

- gnu gcc compiler
- Intel Proccessor
- python3, numpy, pandas, matplotlib(optional)

# How to run

To get started, clone or download this repository and navigate to the src directory:

```bash
cd src
```

There are two files to run: One that shows basic GFLOPS for different loop orderings (`matmulp.c`), and one that implments the GEBP optimization (`goto_van.c`).

### Matmulp.c

To run `matmulp.c`, in the src directory, compule the file with:

```bash
gcc -fopenmp -O3 -march=native  matmulp.c -o matmulp -lm
```
Then run:

```bash
./matmulp
```

### Gotovan.c

To compile, in the src dir, run:

```bash
gcc -O3 -march=native goto_van.c matrix_ops.c -o gotovan -lm
```

And then you can run the program normally with:

```bash
./gotovan
```

Or in debug mode (which prints intermediate matrices):

```bash
./gotovan 1
```

You can visualize the output with:
```bash
python3 visualize.py
```

That will generate a 3d scatter plot using your csv data.

Note: when running in debug mode, matricies are printed to the console. So ensure that these matricies are small enough not to overflow the terminal.

### Output

Each program generates a CSV file located at `../output/` with details about the runtime. The visualizations are also included in the output folder.

# Additional Notes
Before running, you should probably tailor the program to you PC to get accurate utilization and GFLOP calcualtions. For the `gotovan.c`, you can find the piece of code to modify at the top:

```c
#define MAX_FREQ (3.3)
#define CACHE_L1_SIZE (32 * 1024)  // L1 cache size 
#define CACHE_L2_SIZE (double)(256 * 1024)  // L2 cache size
#define ELEMENT_SIZE sizeof(double)  // Size of each matrix element
#define MAX_FREQ (3.97) // Max Clock Frequency of CPU core
#define L1_SIZE_KB (CACHE_L1_SIZE / 1024.00) // L1 cache size kb
#define L2_SIZE_KB (CACHE_L2_SIZE / 1024.00) // L2 cache size kb
#define DEFAULT_N (1024 * 3) // Default dims of matricies

/**
 * Max Clock Frequency of CPU core.
 *  (Frequency in GHz) x (# doubles in AVX size)
 *  x (2 for FMA instruction) x (# of AVX units)
 */
#define MAX_FLOPS (MAX_FREQ * 4 * 2 * 2) // Max Gflops per core of CPU
```
You can find the neccessary values by typing into your terminal `lscpu` or looking up your specs on Intel Ark.

## Program Files:

### `goto_van.c`:
- Matrix multiplication using the GEBP algorithm.
- Outputs performance metrics including GFLOPS and cache usage.

### `matmulp.c`:
- Matrix multiplication using six different loop orderings.
- Outputs performance metrics and writes results to a CSV file.

# Contributors:
- Aman Hogan-Bailey
- [Anatomy of High-Performance Matrix Multiplication by KAZUSHIGE GOTO and ROBERT A. VAN DE GEIJN](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf).
- University of Texas at Arlington