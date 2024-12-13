// task2.cu
#include <iostream>
#include <cmath>
#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <mpi.h>

#include "task2.cuh"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure NZ is divisible by size
    if (NZ % size != 0) {
        if (rank == 0) {
            printf("NZ must be divisible by the number of MPI processes.\n");
        }
        MPI_Finalize();
        return -1;
    }

    // Set the CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(rank % device_count);

    // // Debug for HIP device
    // int device;
    // hipGetDevice(&device);
    // printf("MPI rank %d using device %d\n", rank, device);

    if (rank == 0) {
        printf("Running task2 with %d MPI processes...\n", size);
    }

    // Grid dimensions
    const int nx = NX;
    const int ny = NY;
    const int nz_global = NZ;

    // Compute local nz
    int nz_per_rank = nz_global / size;
    int nz_local = nz_per_rank + 2; // Add 2 for halo layers

    const int local_size = nx * ny * nz_local;

    // Grid spacing
    const double dx = 1.0 / (nx - 1);
    const double dy = 1.0 / (ny - 1);
    const double dz = 1.0 / (nz_global - 1);

    // Allocate memory on host
    double *phi_exact = new double[local_size];

    // Allocate memory on device
    double *d_phi, *d_phi_new, *d_f, *d_phi_exact;
    cudaMalloc((void**)&d_phi, local_size * sizeof(double));
    cudaMalloc((void**)&d_phi_new, local_size * sizeof(double));
    cudaMalloc((void**)&d_f, local_size * sizeof(double));
    cudaMalloc((void**)&d_phi_exact, local_size * sizeof(double));
    cudaCheckError();

    // Initialize phi and phi_new on device
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                   (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
                   nz_local);

    // Compute starting index along z for this rank (excluding halo layers)
    int z_start = rank * nz_per_rank;

    // Initialize phi and phi_new with boundary conditions
    initializePhi<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_new, nx, ny, nz_local, dx, dy, dz, z_start);
    cudaCheckError();

    // Initialize f on device
    initializeF<<<numBlocks, threadsPerBlock>>>(d_f, nx, ny, nz_local, dx, dy, dz, z_start);
    cudaCheckError();

    // Initialize phi_exact on device
    computeExactSolution<<<numBlocks, threadsPerBlock>>>(d_phi_exact, nx, ny, nz_local, dx, dy, dz, z_start);
    cudaCheckError();

    // Copy phi_exact to host for error computation
    cudaMemcpy(phi_exact, d_phi_exact, local_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Allocate arrays for diff, residual, and error
    double *d_diff_array, *d_residual_array, *d_error_array;
    cudaMalloc((void**)&d_diff_array, local_size * sizeof(double));
    cudaMalloc((void**)&d_residual_array, local_size * sizeof(double));
    cudaMalloc((void**)&d_error_array, local_size * sizeof(double));
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

    unsigned long long num_interior = (unsigned long long)(nx - 2) * (ny - 2) * (nz_per_rank);
    unsigned long long flops_per_point = 19;
    unsigned long long bytes_per_point = 80;

    // MPI requests
    MPI_Request requests[4];
    int num_requests;

    // Create custom MPI datatype for halo exchange
    MPI_Datatype xy_plane_type;
    MPI_Type_contiguous(nx * ny, MPI_DOUBLE, &xy_plane_type);
    MPI_Type_commit(&xy_plane_type);

    // Synchronize device before communication
    cudaDeviceSynchronize();

    // Perform initial halo exchange
    num_requests = 0;

    // Use cudaMemcpy to copy halos to host memory for MPI communication
    double *sendbuf_upper = new double[nx * ny];
    double *sendbuf_lower = new double[nx * ny];
    double *recvbuf_upper = new double[nx * ny];
    double *recvbuf_lower = new double[nx * ny];

    // Copy data from device to host
    cudaMemcpy(sendbuf_lower, &d_phi[1 * nx * ny], nx * ny * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(sendbuf_upper, &d_phi[(nz_local - 2) * nx * ny], nx * ny * sizeof(double), cudaMemcpyDeviceToHost);
    cudaCheckError();

    if (rank > 0)
    {
        // Send to lower neighbor
        MPI_Isend(sendbuf_lower, 1, xy_plane_type, rank - 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        // Receive from lower neighbor
        MPI_Irecv(recvbuf_lower, 1, xy_plane_type, rank - 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }

    if (rank < size - 1)
    {
        // Send to upper neighbor
        MPI_Isend(sendbuf_upper, 1, xy_plane_type, rank + 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
        // Receive from upper neighbor
        MPI_Irecv(recvbuf_upper, 1, xy_plane_type, rank + 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }

    // Wait for communications to complete
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);

    // Copy received halos back to device
    if (rank > 0)
    {
        cudaMemcpy(&d_phi[0 * nx * ny], recvbuf_lower, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
    }
    if (rank < size - 1)
    {
        cudaMemcpy(&d_phi[(nz_local - 1) * nx * ny], recvbuf_upper, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaCheckError();

    // Free host buffers
    delete[] sendbuf_lower;
    delete[] sendbuf_upper;
    delete[] recvbuf_lower;
    delete[] recvbuf_upper;

    // Synchronize device before kernel launch
    cudaDeviceSynchronize();

    while (diff > TOLERANCE && iter < MAX_ITER)
    {
        // Synchronize device before communication
        cudaDeviceSynchronize();

        // Perform halo exchange
        num_requests = 0;

        // Use cudaMemcpy to copy halos to host memory for MPI communication
        double *sendbuf_upper = new double[nx * ny];
        double *sendbuf_lower = new double[nx * ny];
        double *recvbuf_upper = new double[nx * ny];
        double *recvbuf_lower = new double[nx * ny];

        // Copy data from device to host
        cudaMemcpy(sendbuf_lower, &d_phi[1 * nx * ny], nx * ny * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sendbuf_upper, &d_phi[(nz_local - 2) * nx * ny], nx * ny * sizeof(double), cudaMemcpyDeviceToHost);
        cudaCheckError();

        if (rank > 0)
        {
            // Receive from lower neighbor
            MPI_Irecv(recvbuf_lower, 1, xy_plane_type, rank - 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
            // Send to lower neighbor
            MPI_Isend(sendbuf_lower, 1, xy_plane_type, rank - 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
        }

        if (rank < size - 1)
        {
            // Receive from upper neighbor
            MPI_Irecv(recvbuf_upper, 1, xy_plane_type, rank + 1, 0, MPI_COMM_WORLD, &requests[num_requests++]);
            // Send to upper neighbor
            MPI_Isend(sendbuf_upper, 1, xy_plane_type, rank + 1, 1, MPI_COMM_WORLD, &requests[num_requests++]);
        }

        // Wait for communications to complete
        MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);

        // Copy received halos back to device
        if (rank > 0)
        {
            cudaMemcpy(&d_phi[0 * nx * ny], recvbuf_lower, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
        }
        if (rank < size - 1)
        {
            cudaMemcpy(&d_phi[(nz_local - 1) * nx * ny], recvbuf_upper, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
        }
        cudaCheckError();

        // Free host buffers
        delete[] sendbuf_lower;
        delete[] sendbuf_upper;
        delete[] recvbuf_lower;
        delete[] recvbuf_upper;

        // Synchronize device before kernel launch
        cudaDeviceSynchronize();

        // Jacobi iteration
        jacobiIteration<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_new, d_f, nx, ny, nz_local, dx, dy, dz, d_diff_array, z_start);
        cudaCheckError();

        // Synchronize device after kernel launch
        cudaDeviceSynchronize();

        // Swap phi and phi_new
        std::swap(d_phi, d_phi_new);

        // Compute local diff using thrust reduction
        thrust::device_ptr<double> diff_ptr = thrust::device_pointer_cast(d_diff_array);
        double local_diff = thrust::reduce(diff_ptr, diff_ptr + local_size, 0.0, thrust::plus<double>());
        cudaCheckError();

        // Compute global diff
        double global_diff;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        diff = sqrt(global_diff);

        iter++;

        // Accumulate FLOPs and bytes
        total_flops += num_interior * flops_per_point;
        total_bytes += num_interior * bytes_per_point;

        if (iter % 100 == 0)
        {
            // Synchronize device before residual computation
            cudaDeviceSynchronize();

            // Compute residual
            computeResidual<<<numBlocks, threadsPerBlock>>>(d_phi, d_f, d_residual_array, nx, ny, nz_local, dx, dy, dz, z_start);
            cudaCheckError();

            // Synchronize device after kernel launch
            cudaDeviceSynchronize();

            // Compute local residual
            thrust::device_ptr<double> res_ptr = thrust::device_pointer_cast(d_residual_array);
            double local_residual = thrust::reduce(res_ptr, res_ptr + local_size, 0.0, thrust::plus<double>());
            cudaCheckError();

            // Compute global residual
            double global_residual;
            MPI_Allreduce(&local_residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double residual = sqrt(global_residual);

            // Compute error
            computeError<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_exact, d_error_array, nx, ny, nz_local, z_start);
            cudaCheckError();

            // Synchronize device after kernel launch
            cudaDeviceSynchronize();

            thrust::device_ptr<double> err_ptr = thrust::device_pointer_cast(d_error_array);
            double local_error = thrust::reduce(err_ptr, err_ptr + local_size, 0.0, thrust::plus<double>());
            cudaCheckError();

            double global_error;
            MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double error = sqrt(global_error);

            if (rank == 0)
            {
                printf("Iteration %d, Residual: %e, Error: %e\n", iter, residual, error);
            }
        }
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float total_time = milliseconds / 1000.0f;

    // Compute final residual and error
    cudaDeviceSynchronize();

    computeResidual<<<numBlocks, threadsPerBlock>>>(d_phi, d_f, d_residual_array, nx, ny, nz_local, dx, dy, dz, z_start);
    cudaCheckError();

    cudaDeviceSynchronize();

    thrust::device_ptr<double> res_ptr = thrust::device_pointer_cast(d_residual_array);
    double local_residual = thrust::reduce(res_ptr, res_ptr + local_size, 0.0, thrust::plus<double>());
    cudaCheckError();

    double global_residual;
    MPI_Allreduce(&local_residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double residual = sqrt(global_residual);

    computeError<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_exact, d_error_array, nx, ny, nz_local, z_start);
    cudaCheckError();

    cudaDeviceSynchronize();

    thrust::device_ptr<double> err_ptr = thrust::device_pointer_cast(d_error_array);
    double local_error = thrust::reduce(err_ptr, err_ptr + local_size, 0.0, thrust::plus<double>());
    cudaCheckError();

    double global_error;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double error = sqrt(global_error);

    unsigned long long global_flops = 0;
    unsigned long long global_bytes = 0;
    
    MPI_Reduce(&total_flops, &global_flops, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_bytes, &global_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        double total_bandwidth = (double)global_bytes / total_time / 1e9; // GB/s
        double total_GFLOPS = (double)global_flops / total_time / 1e9;    // GFLOPS
    
        printf("Final Iteration %d, Residual: %e, Error: %e\n", iter, residual, error);
        printf("Total Time: %f s\n", total_time);
        printf("Total Bandwidth: %f GB/s\n", total_bandwidth);
        printf("Total GFLOPS: %f GFLOPS\n", total_GFLOPS);
    }

    // Free memory
    cudaFree(d_phi);
    cudaFree(d_phi_new);
    cudaFree(d_f);
    cudaFree(d_phi_exact);
    cudaFree(d_diff_array);
    cudaFree(d_residual_array);
    cudaFree(d_error_array);

    delete[] phi_exact;

    MPI_Type_free(&xy_plane_type);

    MPI_Finalize();

    return 0;
}