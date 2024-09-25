# Matrix Multiplication and GEBP Optimization Project

### Summary
This project is part of UTA's Parallel Processing Course (CSE-5351) and focuses on optimizing matrix multiplication performance to achieve maximum GFLOPS. It implements the **Generalized Blocked Panel (GEBP)** algorithm and explores different matrix multiplication techniques, particularly focusing on loop orderings and parameter tuning to maximize computational efficiency.

### Project Components
- **Matrix Multiplication**: Implementations with various loop ordering strategies.
- **GEBP Algorithm**: Tuning the GEBP algorithm for different block sizes to optimize cache usage and maximize performance.
- **Visualization**: A Python script to visualize the performance of the GEBP algorithm in terms of GFLOPS and execution time.

---

### Prerequisites
To run the project, ensure you have the following installed:
- GCC compiler (with OpenMP support)
- Intel processor (for optimized performance)
- Python 3 (optional, for visualization)
  - Numpy
  - Pandas
  - Matplotlib

---

### How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd repo/src/
   ```

2. **Matrix Multiplication with Different Loop Ordering**:
   Compile and run the matrix multiplication implementation with different loop orderings:
   ```bash
   gcc -fopenmp -O0 -march=native -fno-tree-vectorize -mno-fma matmulp.c -o matmulp
   ./matmulp
   ```

3. **GEBP Algorithm with Tunable Parameters**:
   Compile and run the GEBP algorithm with different parameters:
   ```bash
   gcc -fopenmp -O0 -march=native -fno-tree-vectorize -mno-fma goto_van.c -o gotovan -lm
   ./gotovan
   ```

4. **Visualization**:
   Visualize the performance data (GFLOPS, execution time) using the provided Python script:
   ```bash
   python3 visualize.py
   ```

   This script generates graphical and tabular formats of the performance metrics for easy analysis.

---

### Output
All outputs, including performance data and visualizations, will be located in the `output` directory.

---

### Summary of Key Techniques
The project explores several performance optimization techniques, including:
- **Cache Blocking**: Partitioning matrices into blocks that fit into L1 and L2 caches to reduce cache misses.
- **Loop Unrolling**: Manually unrolling loops to reduce loop overhead and improve instruction-level parallelism.
- **Parallelism**: Utilizing OpenMP for multithreading to take full advantage of multi-core processors.
- **GFLOPS Maximization**: Tuning block sizes in the GEBP algorithm to maximize floating-point operations per second (GFLOPS).

# Contributors
- Aman Hogan-Bailey