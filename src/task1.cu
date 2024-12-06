// task1.cu
#ifdef USE_HIP

#include <hip/hip_runtime.h>

// Device management macros
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaSetDevice            hipSetDevice
#define cudaDeviceSynchronize    hipDeviceSynchronize

// Memory management macros
#define cudaMalloc               hipMalloc
#define cudaFree                 hipFree
#define cudaMemcpy               hipMemcpy
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// Error handling macros
#define cudaError_t              hipError_t
#define cudaSuccess              hipSuccess
#define cudaGetLastError         hipGetLastError
#define cudaGetErrorString       hipGetErrorString

// Event management macros
#define cudaEvent_t              hipEvent_t
#define cudaEventCreate          hipEventCreate
#define cudaEventRecord          hipEventRecord
#define cudaEventSynchronize     hipEventSynchronize
#define cudaEventElapsedTime     hipEventElapsedTime
#define cudaEventDestroy         hipEventDestroy

// Stream management macros
#define cudaStream_t             hipStream_t
#define cudaStreamCreate         hipStreamCreate
#define cudaStreamDestroy        hipStreamDestroy
#define cudaStreamSynchronize    hipStreamSynchronize

// Device properties
#define cudaDeviceProp           hipDeviceProp_t
#define cudaGetDeviceProperties  hipGetDeviceProperties

#else

#include <cuda_runtime.h>

#endif

#include <iostream>
#include <cmath>
#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "task1.cuh"

// #define NX 128
// #define NY 128
// #define NZ 128

#define NX 256
#define NY 256
#define NZ 256

// #define NX 384
// #define NY 384
// #define NZ 384

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define TOLERANCE 1e-6
#define MAX_ITER 1000000

// CUDA error checking macro
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                               \
    if(e!=cudaSuccess) {                                            \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,      \
                cudaGetErrorString(e));                             \
        exit(0);                                                    \
    }                                                               \
}

__global__ void initializePhi(double *phi, double *phi_new, int nx, int ny, int nz);
__global__ void initializeF(double *f, int nx, int ny, int nz, double dx, double dy, double dz);
__global__ void jacobiIteration(double *phi, double *phi_new, double *f, int nx, int ny, int nz, double dx, double dy, double dz, double *diff_array);
__global__ void computeResidual(double *phi, double *f, double *residual_array, int nx, int ny, int nz, double dx, double dy, double dz);
__global__ void computeError(double *phi, double *phi_exact, double *error_array, int nx, int ny, int nz);
__global__ void computeExactSolution(double *phi_exact, int nx, int ny, int nz, double dx, double dy, double dz);

int main()
{
    printf("Running task1...\n");

    // Grid dimensions
    const int nx = NX;
    const int ny = NY;
    const int nz = NZ;
    const int size = nx * ny * nz;

    // Grid spacing
    const double dx = 1.0 / (nx - 1);
    const double dy = 1.0 / (ny - 1);
    const double dz = 1.0 / (nz - 1);

    // Allocate memory on host
    double *phi = new double[size];
    double *phi_new = new double[size];
    double *f = new double[size];
    double *phi_exact = new double[size];

    // Allocate memory on device
    double *d_phi, *d_phi_new, *d_f, *d_phi_exact;
    cudaMalloc((void**)&d_phi, size * sizeof(double));
    cudaMalloc((void**)&d_phi_new, size * sizeof(double));
    cudaMalloc((void**)&d_f, size * sizeof(double));
    cudaMalloc((void**)&d_phi_exact, size * sizeof(double));
    cudaCheckError();

    // Initialize phi and phi_new to zero on device
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                   (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
                   nz);

    initializePhi<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_new, nx, ny, nz);
    cudaCheckError();

    // Initialize f on device
    initializeF<<<numBlocks, threadsPerBlock>>>(d_f, nx, ny, nz, dx, dy, dz);
    cudaCheckError();

    // Initialize phi_exact on device
    computeExactSolution<<<numBlocks, threadsPerBlock>>>(d_phi_exact, nx, ny, nz, dx, dy, dz);
    cudaCheckError();

    // Copy phi_exact to host
    cudaMemcpy(phi_exact, d_phi_exact, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Allocate arrays for diff, residual, and error
    double *d_diff_array, *d_residual_array, *d_error_array;
    cudaMalloc((void**)&d_diff_array, size * sizeof(double));
    cudaMalloc((void**)&d_residual_array, size * sizeof(double));
    cudaMalloc((void**)&d_error_array, size * sizeof(double));
    cudaCheckError();

    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Jacobi iteration variables
    double diff = TOLERANCE + 1.0;
    int iter = 0;

    // Performance counters
    unsigned long long total_flops = 0;
    unsigned long long total_bytes = 0;

    unsigned long long num_interior = (unsigned long long)(nx - 2) * (ny - 2) * (nz - 2);
    unsigned long long flops_per_point = 15;
    unsigned long long bytes_per_point = 80;

    while (diff > TOLERANCE && iter < MAX_ITER)
    {
        // Jacobi iteration
        jacobiIteration<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_new, d_f, nx, ny, nz, dx, dy, dz, d_diff_array);
        cudaCheckError();

        // Swap phi and phi_new
        double *temp = d_phi;
        d_phi = d_phi_new;
        d_phi_new = temp;

        // Compute diff using thrust reduction
        thrust::device_ptr<double> diff_ptr = thrust::device_pointer_cast(d_diff_array);
        diff = thrust::reduce(diff_ptr, diff_ptr + size);
        diff = sqrt(diff);

        iter++;

        // Accumulate FLOPs and bytes
        total_flops += num_interior * flops_per_point;
        total_bytes += num_interior * bytes_per_point;

        if (iter % 100 == 0)
        {
            // Compute residual
            computeResidual<<<numBlocks, threadsPerBlock>>>(d_phi, d_f, d_residual_array, nx, ny, nz, dx, dy, dz);
            cudaCheckError();

            // Compute L2 norm of residual
            thrust::device_ptr<double> res_ptr = thrust::device_pointer_cast(d_residual_array);
            double residual = thrust::reduce(res_ptr, res_ptr + size);
            residual = sqrt(residual);

            // Compute error
            computeError<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_exact, d_error_array, nx, ny, nz);
            cudaCheckError();

            thrust::device_ptr<double> err_ptr = thrust::device_pointer_cast(d_error_array);
            double error = thrust::reduce(err_ptr, err_ptr + size);
            error = sqrt(error);

            printf("Iteration %d, Residual: %e, Error: %e\n", iter, residual, error);
        }
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float total_time = milliseconds / 1000.0f;

    // Compute final residual and error
    computeResidual<<<numBlocks, threadsPerBlock>>>(d_phi, d_f, d_residual_array, nx, ny, nz, dx, dy, dz);
    cudaCheckError();

    thrust::device_ptr<double> res_ptr = thrust::device_pointer_cast(d_residual_array);
    double residual = thrust::reduce(res_ptr, res_ptr + size);
    residual = sqrt(residual);

    computeError<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_exact, d_error_array, nx, ny, nz);
    cudaCheckError();

    thrust::device_ptr<double> err_ptr = thrust::device_pointer_cast(d_error_array);
    double error = thrust::reduce(err_ptr, err_ptr + size);
    error = sqrt(error);

    printf("Final Iteration %d, Residual: %e, Error: %e\n", iter, residual, error);

    // Compute total bandwidth and FLOPS
    double total_bandwidth = (double)total_bytes / total_time / 1e9; // GB/s
    double total_GFLOPS = (double)total_flops / total_time / 1e9;    // GFLOPS

    printf("Total Time: %f s\n", total_time);
    printf("Total Bandwidth: %f GB/s\n", total_bandwidth);
    printf("Total GFLOPS: %f GFLOPS\n", total_GFLOPS);

    // Free memory
    cudaFree(d_phi);
    cudaFree(d_phi_new);
    cudaFree(d_f);
    cudaFree(d_phi_exact);
    cudaFree(d_diff_array);
    cudaFree(d_residual_array);
    cudaFree(d_error_array);

    delete[] phi;
    delete[] phi_new;
    delete[] f;
    delete[] phi_exact;

    return 0;
}
