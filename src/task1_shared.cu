// task1.cu
#include <iostream>
#include <cmath>
#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "task1.cuh"

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

    // Pinned host memory
    double *phi, *phi_new, *phi_exact, *f;
    cudaHostAlloc((void**)&phi,       size * sizeof(double), 0);
    cudaHostAlloc((void**)&phi_new,   size * sizeof(double), 0);
    cudaHostAlloc((void**)&phi_exact, size * sizeof(double), 0);
    cudaHostAlloc((void**)&f,         size * sizeof(double), 0);

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
    cudaMemcpyAsync(phi_exact, d_phi_exact, size * sizeof(double), cudaMemcpyDeviceToHost);
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
    unsigned long long flops_per_point = 19;
    unsigned long long bytes_per_point = 80;

    while (diff > TOLERANCE && iter < MAX_ITER)
    {
        // Jacobi iteration with shared memory
        // jacobiIteration<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_new, d_f, nx, ny, nz, dx, dy, dz, d_diff_array);
        size_t shmem_size = (BLOCK_SIZE_X+2)*(BLOCK_SIZE_Y+2)*sizeof(double);
        jacobiIterationShared<<<numBlocks, threadsPerBlock, shmem_size>>>(d_phi, d_phi_new, d_f, nx, ny, nz, dx, dy, dz, d_diff_array);
        cudaCheckError();

        // Swap phi and phi_new
        std::swap(d_phi, d_phi_new);

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

    // Free device memory
    cudaFree(d_phi);
    cudaFree(d_phi_new);
    cudaFree(d_f);
    cudaFree(d_phi_exact);
    cudaFree(d_diff_array);
    cudaFree(d_residual_array);
    cudaFree(d_error_array);

    // Free host memory
    cudaFreeHost(phi);
    cudaFreeHost(phi_new);
    cudaFreeHost(phi_exact);
    cudaFreeHost(f);

    return 0;
}
