 /**
 * Author: Aman Hogan-Bailey
 * Uses GEBP optimization to 
 * perform matrix multiplication with
 * differnt paramater sizes. Everything 
 * is accessed and stored in column major order.
 */

#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>
#include "matrix_ops.h"

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

/**
 * ## GEBP Algorithm
 * 
 * Performs a blocked matrix multiplication using the Generalized Blocked Panel algorithm. 
 * Matrices A, B, and C are stored in column-major order. The function breaks the 
 * matrices into smaller blocks that fit into the L1 and L2 caches.
 * @param N The dimension of A,B,C
 * @param A Matrix A
 * @param B Matrix B
 * @param C Matrix C
 * @param m_c Number of rows of A and C to process at one time
 * @param k_c How many columns of A and rows of B to process at a time.
 * @param n_r Number of columns of B and C to process at a time.
 * @param m_r Number of rows within A and C that fits into registers.
 */
void gebp(int N, double* A, double* B, double* C, int m_c, int k_c, int n_r, int m_r) 
{
    
    double* A_block = (double*)malloc(m_c * k_c * sizeof(double)); // Block of A: m_c x k_c
    double* B_sliver = (double*)malloc(k_c * n_r * sizeof(double)); // Block of B: k_c x n_r
    double* C_block = (double*)malloc(m_c * n_r * sizeof(double)); // Block of C: m_c x n_r 

    // Loop over the i dimension (rows of A and C)
    for (int i_block = 0; i_block < N; i_block += m_c) 
    {
        // Loop over the k dimension (columns of A and rows of B)
        for (int k_block = 0; k_block < N; k_block += k_c) 
        {
            load_A(A_block, A, N, m_c, k_c, i_block, k_block);

            // Loop over the j dimension (columns of B and C)
            for (int j_block = 0; j_block < N; j_block += n_r) 
            {
                load_B(B_sliver, B, N, k_c, n_r, j_block, k_block);
                load_C(C_block, C, N, m_c, n_r, i_block, j_block);
                multiply_blocks_avx(A_block, B_sliver, C_block, m_c, k_c, n_r, m_r);
                store_C(C_block, C, N, m_c, n_r, i_block, j_block);
            }
        }
    }

    free(A_block);
    free(B_sliver);
    free(C_block);
}


int main(int argc, char* argv[]) 
{
    int debug = 0;
    int N = DEFAULT_N;

    double* A = (double*)malloc(N * N * sizeof(double)); // A matrix
    double* B = (double*)malloc(N * N * sizeof(double)); // B matrix
    double* C = (double*)malloc(N * N * sizeof(double)); // C matrix

    // Initialize Matricies
    for (int i = 0; i < N * N; i++) 
    {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
        C[i] = rand() % 10;
    }

    // Create csv file for table
    FILE* fp = fopen("../output/gotovan.csv", "w"); 
    if (fp == NULL)
    {
        perror("Unable to open file for writing.");
        return 1;
    }

    // Check if in debug mode or not
    if (argc > 1)
    { 
        debug = atoi(argv[1]);
    }

    int kc_sizes[] = {N/32, N/30, N/28, N/24, N/20}; // Different sizes of columns of block A
    int mr_sizes[] = {4, 8, 16, 32, 96}; // Different sizes of register blocking batches
    int nr_sizes[] = {4, 8, 16};  // Different sizes of number of rows in Block B

    printf("Debug mode: %s\n", debug ? "ON" : "OFF");
    fprintf(fp, "kc/mc,nr,mr,gflops,time (seconds),util, A block (KB), B Sliver (KB)\n");

    // Loop over different k_c, n_r, and m_r sizes
    for (int kc_index = 0; kc_index < sizeof(kc_sizes)/sizeof(kc_sizes[0]); kc_index++)
    {
        for (int mr_index = 0; mr_index < sizeof(mr_sizes)/sizeof(mr_sizes[0]); mr_index++)
        {
            for (int nr_index = 0; nr_index < sizeof(nr_sizes)/sizeof(nr_sizes[0]); nr_index++)
            {
                
                printf("---------------------------------------\n");
                int k_c = kc_sizes[kc_index]; // columns in block a 
                int m_c = k_c; // of rows in block A and C
                int m_r = mr_sizes[mr_index]; // of registers for register blocking
                int n_r = nr_sizes[nr_index]; // of columns in block B and C

                if (m_r > k_c || m_r == m_c/2) 
                { 
                    printf("m_r cannot exceed the size of k_c.");
                    printf("Nor can m_r = k_c/2. Choose a different size of N or change size of m_r.\n");
                    return 1;
                }

                // Reset C matrix to 0 before each run
                for (int i = 0; i < N * N; i++) 
                {
                    C[i] = 0.0;
                }

                if (debug == 1) {print_matrix(A, N,N, "A");}
                if (debug == 1) {print_matrix(B, N,N, "B");}

                printf("Matrix Sizes: %dx%d\n", N,N);
                printf("Block Sizes: m_c = %d, k_c = %d, n_r = %d, m_r = %d\n", m_c, k_c, n_r, m_r);

                struct timespec start, end;
                clock_gettime(CLOCK_MONOTONIC, &start);
                gebp(N, A, B, C, m_c, k_c, n_r, m_r);
                clock_gettime(CLOCK_MONOTONIC, &end);

                double delta_t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; // runtime of algo
                double g_flops = ((double) N * N * N * 2 / delta_t) / 1e9; // gigflops of algorithm
                double util = g_flops / MAX_FLOPS; // utilization of algo

                int l1_used = (n_r * m_c * 2 * 8) /1024; // space taken in l1 cache kb
                int l2_used = (k_c * m_c * 8) /1024; // space taken in l2 cache kb

                print_performance_info(delta_t,g_flops,util,l1_used,l2_used,L1_SIZE_KB,L2_SIZE_KB,k_c,n_r,m_r,fp);

                if (debug == 1) {print_matrix(C, N,N, "C");}
            }
        }
    }

    free(A);
    free(B);
    free(C);
    fclose(fp);
    return 0;
}