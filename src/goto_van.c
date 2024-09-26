/**
 * Author: Aman Hogan-Bailey
 * Uses GEBP optimization to 
 * perform matrix multiplication with
 * differnt paramater sizes.
 */

#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>


#define CACHE_L1_SIZE (192 * 1024)  // L1 cache size 
#define CACHE_L2_SIZE (double)(1.5 * 1024 * 1024)  // L2 cache size
#define ELEMENT_SIZE sizeof(double)  // Size of each matrix element
#define MAX_FREQ (3.97) // Max Clock Frequency of CPU core
#define MAX_FLOPS (MAX_FREQ * 4 * 2 * 2) // Max Gflops per core of CPU
#define CACHE_L1_SIZE_KB (CACHE_L1_SIZE / 1024.00) // L1 cache size kb
#define CACHE_L2_SIZE_KB (CACHE_L2_SIZE / 1024.00) // L2 cache size kb

/**
 * Performs matrix multiplication for blocks of A and B, 
 * and puts the result in C block using column-major order.
 * @param A_block block of A in memory (size: m_r x k_c)
 * @param B_sliver sliver of B in memory (size: k_c x n_r)
 * @param C_block block of C in memory (size: m_r x n_r)
 * @param m_r number of rows in A_block and C_block
 * @param n_r number of columns in B_sliver and C_block
 * @param k_c the number of columns in A_block and rows in B_sliver
 */
void multiply_block(double* A_block, double* B_sliver, double* C_block, int m_r, int n_r, int k_c) 
{
    // Iterate over the columns of B_sliver and C_block
    for (int j = 0; j < n_r; j++) 
    {  
        // Iterate over the rows of A_block and C_block
        for (int i = 0; i < m_r; i++) 
        {  
            // dot product of the i-th row of A_block and j-th column of B_sliver
            double sum = 0.0;
            for (int k = 0; k < k_c; k++) 
            {  
                sum += A_block[k * m_r + i] * B_sliver[j * k_c + k];
            }
            C_block[j * m_r + i] += sum;
        }
    }
}

/**
 * Performs a blocked matrix multiplication using the Generalized Blocked Panel algorithm. 
 * where matrices A, B, and C are stored in column-major order. 
 * The function breaks the matrices into smaller blocks that fit into the L1 and L2 caches.
 * @param N The dimension of the square matrices
 * @param A Pointer to the matrix A
 * @param B Pointer to the matrix B
 * @param C Pointer to the matrix C
 * @param m_c Number of rows of A and C to process at one time
 * @param k_c Hhow many columns of A and rows of B to process at a time.
 * @param n_r Number of columns of B and C to process at a time.
 * @param m_r Size of the block of rows within A and C that fits into registers.
 */
void gebp(int N, double* A, double* B, double* C, int m_c, int k_c, int n_r, int m_r) {
    
    // Block of A
    double* A_block = (double*)malloc(m_c * k_c * sizeof(double));

    // Block of B
    double* B_sliver = (double*)malloc(k_c * n_r * sizeof(double));

    // Block of C
    double* C_block = (double*)malloc(m_r * n_r * sizeof(double));

    // Loop over the i dimension (rows of A and C), processing m_c rows of A and C at a time.
    for (int i_block = 0; i_block < N; i_block += m_c) 
    {

        // Loop over the k dimension (columns of A and rows of B)
        for (int k_block = 0; k_block < N; k_block += k_c) 
        {
            
            // Load the block of A (size m_c x k_c)
            for (int k = 0; k < k_c && k_block + k < N; k++) 
            {
                for (int i = 0; i < m_c && i_block + i < N; i++) 
                {
                    A_block[k * m_c + i] = A[(k_block + k) * N + (i_block + i)]; 
                }
            }

            // Loop over the j dimension (columns of B and C)
            for (int j_block = 0; j_block < N; j_block += n_r) 
            {
                
                // Load a sliver of B (size k_c x n_r) 
                for (int j = 0; j < n_r && j_block + j < N; j++)
                 {
                    for (int k = 0; k < k_c && k_block + k < N; k++) 
                    {
                        B_sliver[j * k_c + k] = B[(k_block + k) * N + (j_block + j)];
                    }
                }

                // Load the block of C (size m_r x n_r) 
                for (int j = 0; j < n_r && j_block + j < N; j++)
                {
                    for (int i = 0; i < m_r && i_block + i < N; i++) 
                    {
                        C_block[j * m_r + i] = C[(j_block + j) * N + (i_block + i)];
                    }
                }

                // Block multiplication: C_block += A_block * B_sliver
                multiply_block(A_block, B_sliver, C_block, m_r, n_r, k_c);

                // Store the result of C_block (column-major order).
                for (int j = 0; j < n_r && j_block + j < N; j++)
                {
                    for (int i = 0; i < m_r && i_block + i < N; i++)
                    {
                        C[(j_block + j) * N + (i_block + i)] = C_block[j * m_r + i]; 
                    }
                }
            }
        }
    }

    free(A_block);
    free(B_sliver);
    free(C_block);
}

int main() 
{

    int kc_sizes[] = {4, 8, 16};  
    int mr_sizes[] = {2, 4, 8};  
    int nr_sizes[] = {2, 4, 8}; 

    // Dimension of square matricies
    int N = 512 * 6;

    double* A = (double*)malloc(N * N * sizeof(double)); // A matrix
    double* B = (double*)malloc(N * N * sizeof(double)); // B matrix
    double* C = (double*)malloc(N * N * sizeof(double)); // C matrix

    // Init matrices
    for (int i = 0; i < N * N; i++) 
    {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
        C[i] = 0.0;
    }


    // Open csv file
    FILE* fp = fopen("../output/gotovan.csv", "w"); 
    if (fp == NULL)
    {
        perror("Unable to open file for writing.");
        return 1;
    }
    fprintf(fp, "kc/mc,nr,mr,gflops,time (seconds),util\n");

    // Loop over different kc, nr, and mr sizes
    for (int kc_index = 0; kc_index < sizeof(kc_sizes)/sizeof(kc_sizes[0]); kc_index++)
    {
        for (int mr_index = 0; mr_index < sizeof(mr_sizes)/sizeof(mr_sizes[0]); mr_index++)
        {
            for (int nr_index = 0; nr_index < sizeof(nr_sizes)/sizeof(nr_sizes[0]); nr_index++)
            {
                // Calculate optimal Kc block size
                printf("---------------------------------------\n");
                int k_c = (int) sqrt((CACHE_L2_SIZE / 2) / ELEMENT_SIZE);
                k_c = (k_c / 16) * kc_sizes[kc_index];
                int m_c = k_c;

                int m_r = mr_sizes[mr_index];
                int n_r = nr_sizes[nr_index];

                printf("Matrix Size (N): %d\n", N);
                printf("Testing with Block Sizes: m_c = %d, k_c = %d, n_r = %d, m_r = %d\n", m_c, k_c, n_r, m_r);

                // Reset C matrix to 0 before each run
                for (int i = 0; i < N * N; i++) {C[i] = 0.0;}

                struct timespec start, end;
                clock_gettime(CLOCK_MONOTONIC, &start);
                gebp(N, A, B, C, m_c, k_c, n_r, m_r);
                clock_gettime(CLOCK_MONOTONIC, &end);

                // Compute time taken
                double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
                printf("Time taken: %f seconds\n", time_taken);

                // Approximate GFLOPS
                double gflops = ((double) N * N * N * 2 / time_taken) / 1e9;
                
                //if (gflops > MAX_FLOPS) {gflops = MAX_FLOPS;}
                printf("GFLOPS: %lf\n", gflops);

                // Approximate GFLOPS utilization
                double gflops_util = gflops / MAX_FLOPS;
                //if (gflops_util > 1) {gflops_util = .99;}
                printf("GFLOPS Utilization: %lf\n", gflops_util);
                fprintf(fp, "%d,%d,%d,%lf,%f,%lf\n", k_c, n_r, m_r, gflops, time_taken, gflops_util);

                int cache_l1_used = (n_r * k_c * 8) /1024;
                printf("B block size (KB)/L1 Cache size: %d/%lf\n", cache_l1_used, CACHE_L1_SIZE_KB);

                int cache_l2_used = (k_c * m_c * 8) /1024;
                printf("A block size (KB)/L2 Cache size: %d/%lf\n\n", cache_l2_used, CACHE_L2_SIZE_KB);
                printf("---------------------------------------\n");
                
            }
        }
    }

    free(A);
    free(B);
    free(C);
    fclose(fp);
    return 0;
}