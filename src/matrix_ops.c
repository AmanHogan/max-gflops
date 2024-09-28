#include "matrix_ops.h"
#include <stdio.h>
#include <stdlib.h>


void load_A(double* A_block, double* A, int N, int m_c, int k_c, int i_block, int k_block) 
{
    for (int k = 0; k < k_c && k_block + k < N; k++) 
    {
        for (int i = 0; i < m_c && i_block + i < N; i++) 
        {
            A_block[k * m_c + i] = A[(k_block + k) * N + (i_block + i)];
        }
    }
}

void load_B(double* B_sliver, double* B, int N, int k_c, int n_r, int j_block, int k_block) 
{
    for (int j = 0; j < n_r && j_block + j < N; j++) 
    {
        for (int k = 0; k < k_c && k_block + k < N; k++) 
        {
            B_sliver[j * k_c + k] = B[(j_block + j) * N + (k_block + k)];
        }
    }
}

void load_C(double* C_block, double* C, int N, int m_c, int n_r, int i_block, int j_block) 
{
    for (int j = 0; j < n_r && j_block + j < N; j++) 
    {
        for (int i = 0; i < m_c && i_block + i < N; i++) 
        {
            C_block[j * m_c + i] = C[(j_block + j) * N + (i_block + i)];
        }
    }
}

void store_C(double* C_block, double* C, int N, int m_c, int n_r, int i_block, int j_block) 
{
    for (int j = 0; j < n_r; j++) 
    {
        for (int i = 0; i < m_c; i++) 
        {
            C[(j_block + j) * N + (i_block + i)] = C_block[j * m_c + i];
            
        }
    }
}

void print_matrix(double* matrix, int rows, int cols, const char* name)
{
    printf("Matrix %s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            printf("%6.2f ", matrix[j * rows + i]);
        }
        printf("\n");
    }
    printf("\n");
}

void multiply_blocks_avx(double* A_block, double* B_sliver, double* C_block, int m_c, int k_c, int n_r, int m_r)
 {
    for (int j = 0; j < n_r; j++) 
    {
        for (int k = 0; k < k_c; k++) 
        {
            double b_val = B_sliver[j * k_c + k];

            int i;
            for (i = 0; i <= m_c - m_r; i += m_r) 
            {
                __m256d a_val_avx_0 = _mm256_loadu_pd(&A_block[k * m_c + i]);
                __m256d c_val_avx_0 = _mm256_loadu_pd(&C_block[j * m_c + i]);
                __m256d b_val_avx = _mm256_set1_pd(b_val);
                c_val_avx_0 = _mm256_fmadd_pd(a_val_avx_0, b_val_avx, c_val_avx_0);
                _mm256_storeu_pd(&C_block[j * m_c + i], c_val_avx_0);

            }

            for (; i < m_c; i++) 
            {
                C_block[j * m_c + i] += A_block[k * m_c + i] * b_val;

            }
        }
    }
}

void print_performance_info(
    double time_taken, double gflops, double gflops_util,
    int cache_l1_used, int cache_l2_used, 
    double CACHE_L1_SIZE_KB, double CACHE_L2_SIZE_KB, 
    int k_c, int n_r, int m_r,
    FILE* fp
) 
{
    printf("Time taken: %f seconds\n", time_taken);
    printf("GFLOPS: %lf\n", gflops);
    printf("GFLOPS Utilization: %lf\n", gflops_util);
    printf("B block size (KB)/L1 Cache size: %d/%lf\n", cache_l1_used, CACHE_L1_SIZE_KB);
    printf("A block size (KB)/L2 Cache size: %d/%lf\n", cache_l2_used, CACHE_L2_SIZE_KB);
    printf("---------------------------------------\n");

    fprintf(fp, "%d,%d,%d,%lf,%f,%lf,%d/%lf,%d/%lf\n",
                k_c, n_r, m_r, gflops, time_taken, gflops_util,
                cache_l2_used, CACHE_L2_SIZE_KB, cache_l1_used, CACHE_L1_SIZE_KB
            );
}