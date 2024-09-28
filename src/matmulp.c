/**
 * Author: Amna Hogan-Bailey
 * Uta - Parallel proccesing
 * Performing matrix multiplication using different
 * loop orderings and tiems them. 
 * gcc -fopenmp -O3 -march=native  matmulp.c -o matmulp -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matmul_ijk(int N, double** A, double** B, double** C);
void matmul_ikj(int N, double** A, double** B, double** C);
void matmul_kij(int N, double** A, double** B, double** C);
void matmul_kji(int N, double** A, double** B, double** C);
void matmul_jki(int N, double** A, double** B, double** C);
void matmul_jik(int N, double** A, double** B, double** C);

void reset_matrix(int N, double** C);
void free_memory(int N, double** A, double** B, double** C);
int main() 
{
    // Length and width of all matrices
    int N = 512 * 2;

    // A Matrix
    double** A = (double**)malloc(N * sizeof(double*));

    // B Matrix
    double** B = (double**)malloc(N * sizeof(double*));
    
    // Resultant Matrix
    double** C = (double**)malloc(N * sizeof(double*));
    
    // Allocate space for matrices
    for (int i = 0; i < N; i++) 
    {
        A[i] = (double*)malloc(N * sizeof(double));
        B[i] = (double*)malloc(N * sizeof(double));
        C[i] = (double*)malloc(N * sizeof(double));
    }

    // Initialize matrices
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
            C[i][j] = 0.0;
        }
    }

    struct timespec start, end;
    double time_taken;
    double flops, gflops;

    // Open CSV file for writing
    FILE *fp = fopen("../output/matmulp_results.csv", "w");
    if (fp == NULL)
    {
        fprintf(stderr, "Error opening file!\n");
        return 1;
    }
    
    // Write header to CSV file
    fprintf(fp, "Order,Seconds,GFLOPS\n");

    // Measure time for each permutation
    printf("Matrix Multiplications:\n");

    // ijk
    reset_matrix(N, C);
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_ijk(N, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    flops = ((double)N * N * N * 2) / time_taken;
    gflops = flops / 1e9;
    printf("ijk order: %f seconds. GFLOPS: %lf\n", time_taken, gflops);
    fprintf(fp, "ijk,%f,%lf\n", time_taken, gflops);

    // ikj
    reset_matrix(N, C);
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_ikj(N, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    flops = ((double)N * N * N * 2) / time_taken;
    gflops = flops / 1e9;
    printf("ikj order: %f seconds. GFLOPS: %lf\n", time_taken, gflops);
    fprintf(fp, "ikj,%f,%lf\n", time_taken, gflops);

    // kij
    reset_matrix(N, C);
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_kij(N, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    flops = ((double)N * N * N * 2) / time_taken;
    gflops = flops / 1e9;
    printf("kij order: %f seconds. GFLOPS: %lf\n", time_taken, gflops);
    fprintf(fp, "kij,%f,%lf\n", time_taken, gflops);

    // kji
    reset_matrix(N, C);
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_kji(N, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    flops = ((double)N * N * N * 2) / time_taken;
    gflops = flops / 1e9;
    printf("kji order: %f seconds. GFLOPS: %lf\n", time_taken, gflops);
    fprintf(fp, "kji,%f,%lf\n", time_taken, gflops);

    // jki
    reset_matrix(N, C);
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_jki(N, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    flops = ((double)N * N * N * 2) / time_taken;
    gflops = flops / 1e9;
    printf("jki order: %f seconds. GFLOPS: %lf\n", time_taken, gflops);
    fprintf(fp, "jki,%f,%lf\n", time_taken, gflops);

    // jik
    reset_matrix(N, C);
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_jik(N, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    flops = ((double)N * N * N * 2) / time_taken;
    gflops = flops / 1e9;
    printf("jik order: %f seconds. GFLOPS: %lf\n", time_taken, gflops);
    fprintf(fp, "jik,%f,%lf\n", time_taken, gflops);

    // Close the CSV file
    fclose(fp);

    // Free allocated memory
    free_memory(N, A, B, C);
    return 0;
}

void reset_matrix(int N, double** C) 
{
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            C[i][j] = 0.0;
        }
    }
}

void free_memory(int N, double** A, double** B, double** C)
{
    for (int i = 0; i < N; i++) 
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
}

void matmul_ijk(int N, double** A, double** B, double** C) 
{
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) 
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matmul_ikj(int N, double** A, double** B, double** C) 
{
    for (int i = 0; i < N; i++) 
    {
        for (int k = 0; k < N; k++) 
        {
            for (int j = 0; j < N; j++) 
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matmul_kij(int N, double** A, double** B, double** C) 
{
    for (int k = 0; k < N; k++) 
    {
        for (int i = 0; i < N; i++) 
        {
            for (int j = 0; j < N; j++) 
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matmul_kji(int N, double** A, double** B, double** C) 
{
    for (int k = 0; k < N; k++) 
    {
        for (int j = 0; j < N; j++) 
        {
            for (int i = 0; i < N; i++) 
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matmul_jki(int N, double** A, double** B, double** C) 
{
    for (int j = 0; j < N; j++) 
    {
        for (int k = 0; k < N; k++) 
        {
            for (int i = 0; i < N; i++) 
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matmul_jik(int N, double** A, double** B, double** C)
 {
    for (int j = 0; j < N; j++) 
    {
        for (int i = 0; i < N; i++) 
        {
            for (int k = 0; k < N; k++) 
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
