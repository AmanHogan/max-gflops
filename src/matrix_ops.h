#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * This function loads a sub-block of matrix A into A_block, where A is stored
 * in column-major order. The sub-block loaded is determined by the block coordinates
 *
 * @param A_block Pointer to the block of A 
 * @param A Pointer to the original matrix A 
 * @param N The dimension
 * @param m_c Number of rows 
 * @param k_c Number of columns 
 * @param i_block The starting row index for the block in matrix A.
 * @param k_block The starting column index for the block in matrix A.
 */
void load_A(double* A_block, double* A, int N, int m_c, int k_c, int i_block, int k_block);

/**
 * This function loads a sub-sliver of matrix B into B_sliver, where B is stored
 * in column-major order. The sub-sliver loaded is determined by the block coordinates
 * (j_block, k_block).
 *
 * @param B_sliver Pointer to the sliver of B
 * @param B Pointer to matrix B 
 * @param N The dimension
 * @param k_c Number of rows 
 * @param n_r Number of columns 
 * @param j_block The starting column index for the sliver in matrix B.
 * @param k_block The starting row index for the sliver in matrix B.
 */
void load_B(double* B_sliver, double* B, int N, int k_c, int n_r, int j_block, int k_block);

/**
 * This function loads a sub-block of matrix C into C_block, where C is stored
 * in column-major order. The sub-block loaded is determined by the block coordinates
 * (i_block, j_block).
 *
 * @param C_block Pointer to the block of C 
 * @param C Pointer to the matrix C 
 * @param N  The dimension
 * @param m_c Number of rows 
 * @param n_r Number of columns 
 * @param i_block The starting row index for the block in matrix C.
 * @param j_block The starting column index for the block in matrix C.
 */
void load_C(double* C_block, double* C, int N, int m_c, int n_r, int i_block, int j_block);

/**
 * This function stores a sub-block of matrix C from C_block back into C,
 * where C is stored in column-major order. The sub-block is stored at 
 * (i_block, j_block).
 *
 * @param C_block Pointer to the block of C that will be stored back into matrix C.
 * @param C Pointer to the original matrix C stored in column-major order.
 * @param N The dimension of the square matrix C (N x N).
 * @param m_c Number of rows in the block.
 * @param n_r Number of columns in the block.
 * @param i_block The starting row index for the block in matrix C.
 * @param j_block The starting column index for the block in matrix C.
 */
void store_C(double* C_block, double* C, int N, int m_c, int n_r, int i_block, int j_block);

/**
 * This function multiplies a block of matrix A (A_block) with a sliver of matrix B (B_sliver)
 * and accumulates the result in the corresponding block of matrix C (C_block). AVX2 SIMD 
 * intrinsics are used to optimize the matrix multiplication.
 *
 * @param A_block Pointer to the block of A.
 * @param B_sliver Pointer to the sliver of B.
 * @param C_block Pointer to the block of C
 * @param m_c Number of rows in the block of matrix A and C.
 * @param k_c Number of columns in the block of matrix A and rows  B.
 * @param n_r Number of columns in B and matrix C.
 * @param m_r number of rows to be processed at a time within A_block and C_block.
 */
void multiply_blocks_avx(double* A_block, double* B_sliver, double* C_block, int m_c, int k_c, int n_r, int m_r);

/**
 * This function prints the matrix with the given number of rows and columns.
 * The matrix is assumed to be stored in column-major order.
 *
 * @param matrix Pointer to the matrix to be printed.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @param name The name of the matrix to be printed.
 */
void print_matrix(double* matrix, int rows, int cols, const char* name);

/**
 * This function prints and logs various performance metrics
 *
 * @param time_taken Time taken
 * @param gflops Calculated GFLOPS 
 * @param gflops_util Utilization
 * @param cache_l1_used Amount of L1 cache used in KB.
 * @param cache_l2_used Amount of L2 cache used in KB.
 * @param CACHE_L1_SIZE_KB Total size of the L1 cache in KB.
 * @param CACHE_L2_SIZE_KB Total size of the L2 cache in KB.
 * @param k_c columns of A, rows of B
 * @param n_r columns of B and C
 * @param m_r rows of A and C
 * @param fp File pointer to log the performance data into a CSV file.
 */
void print_performance_info(
    double time_taken, double gflops, double gflops_util,
    int cache_l1_used, int cache_l2_used, 
    double CACHE_L1_SIZE_KB, double CACHE_L2_SIZE_KB, 
    int k_c, int n_r, int m_r,
    FILE* fp
);

#endif