// task2.cuh
#ifndef TASK2_CUH
#define TASK2_CUH

#pragma once

#ifdef USE_HIP

#include <hip/hip_runtime.h>

// Device management macros
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaSetDevice            hipSetDevice
#define cudaDeviceSynchronize    hipDeviceSynchronize

// Memory management macros
#define cudaMalloc               hipMalloc
#define cudaHostAlloc            hipHostAlloc
#define cudaFree                 hipFree
#define cudaFreeHost             hipFreeHost
#define cudaMemcpy               hipMemcpy
#define cudaMemcpyAsync          hipMemcpyAsync
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

// CUDA error checking macro
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                               \
    if(e!=cudaSuccess) {                                            \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,      \
                cudaGetErrorString(e));                             \
        exit(0);                                                    \
    }                                                               \
}

#else

#include <cuda_runtime.h>

#endif

#define PI 3.14159265358979323846
#define N_PARAM 2
#define M_PARAM 2
#define K_PARAM 2

#define NX 128
#define NY 128
#define NZ 128

// #define NX 256
// #define NY 256
// #define NZ 256

// #define NX 384
// #define NY 384
// #define NZ 384

// #define BLOCK_SIZE_X 16
// #define BLOCK_SIZE_Y 16

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

#define TOLERANCE 1e-6
#define MAX_ITER 1000000

__global__ void initializePhi(double *phi, double *phi_new, int nx, int ny, int nz, double dx, double dy, double dz, int z_start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index
    int k = blockIdx.z;                            // local z index

    if (i < nx && j < ny && k < nz)
    {
        int global_k = z_start + k - 1; // Adjust for halo

        int idx = i + j * nx + k * nx * ny;

        if (global_k >= 0 && global_k < NZ)
        {
            // Apply Dirichlet boundary conditions at boundaries
            if (i == 0 || i == nx -1 || j == 0 || j == ny -1 || global_k == 0 || global_k == NZ -1)
            {
                double x = i * dx;
                double y = j * dy;
                double z = global_k * dz;

                phi[idx] = sin(N_PARAM * PI * x) * cos(M_PARAM * PI * y) * sin(K_PARAM * PI * z);
                phi_new[idx] = phi[idx];
            }
            else
            {
                phi[idx] = 0.0;
                phi_new[idx] = 0.0;
            }
        }
    }
}

__global__ void initializeF(double *f, int nx, int ny, int nz, double dx, double dy, double dz, int z_start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (i < nx && j < ny && k < nz)
    {
        int global_k = z_start + k - 1;

        if (global_k >= 0 && global_k < NZ)
        {
            double x = i * dx;
            double y = j * dy;
            double z = global_k * dz;

            int idx = i + j * nx + k * nx * ny;

            double phi_val = sin(N_PARAM * PI * x) * cos(M_PARAM * PI * y) * sin(K_PARAM * PI * z);
            f[idx] = - (N_PARAM * N_PARAM + M_PARAM * M_PARAM + K_PARAM * K_PARAM) * PI * PI * phi_val;
        }
    }
}

__global__ void computeExactSolution(double *phi_exact, int nx, int ny, int nz, double dx, double dy, double dz, int z_start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (i < nx && j < ny && k < nz)
    {
        int global_k = z_start + k - 1;

        if (global_k >= 0 && global_k < NZ)
        {
            double x = i * dx;
            double y = j * dy;
            double z = global_k * dz;

            int idx = i + j * nx + k * nx * ny;

            phi_exact[idx] = sin(N_PARAM * PI * x) * cos(M_PARAM * PI * y) * sin(K_PARAM * PI * z);
        }
    }
}

__global__ void jacobiIteration(double *phi, double *phi_new, double *f, int nx, int ny, int nz, double dx, double dy, double dz, double *diff_array, int z_start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (i < nx && j < ny && k < nz)
    {
        int global_k = z_start + k - 1;
        int idx = i + j * nx + k * nx * ny;

        // Handle global boundary conditions
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || global_k == 0 || global_k == NZ - 1)
        {
            double x = i * dx;
            double y = j * dy;
            double z = global_k * dz;
            phi_new[idx] = sin(N_PARAM * PI * x) * cos(M_PARAM * PI * y) * sin(K_PARAM * PI * z);
            diff_array[idx] = 0.0;
        }
        else if (k >= 1 && k < nz - 1)
        {
            int idx_xm = idx - 1;
            int idx_xp = idx + 1;
            int idx_ym = idx - nx;
            int idx_yp = idx + nx;
            int idx_zm = idx - nx * ny;
            int idx_zp = idx + nx * ny;

            double phi_xm = phi[idx_xm];
            double phi_xp = phi[idx_xp];
            double phi_ym = phi[idx_ym];
            double phi_yp = phi[idx_yp];
            double phi_zm = phi[idx_zm];
            double phi_zp = phi[idx_zp];

            double phi_old = phi[idx];

            phi_new[idx] = ((phi_xm + phi_xp) / (dx * dx) +
                            (phi_ym + phi_yp) / (dy * dy) +
                            (phi_zm + phi_zp) / (dz * dz) -
                            f[idx]) /
                           (2.0 / (dx * dx) + 2.0 / (dy * dy) + 2.0 / (dz * dz));

            double diff_local = phi_new[idx] - phi_old;
            diff_array[idx] = diff_local * diff_local;
        }
        else
        {
            phi_new[idx] = phi[idx];
            diff_array[idx] = 0.0;
        }
    }
}

__global__ void computeResidual(double *phi, double *f, double *residual_array, int nx, int ny, int nz, double dx, double dy, double dz, int z_start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (i >= 1 && i < nx - 1 && j >=1 && j < ny - 1 && k >=1 && k < nz -1)
    {
        int idx = i + j * nx + k * nx * ny;

        int global_k = z_start + k - 1;

        if (global_k == 0 || global_k == NZ - 1)
        {
            residual_array[idx] = 0.0;
            return;
        }

        int idx_xm = idx - 1;
        int idx_xp = idx + 1;
        int idx_ym = idx - nx;
        int idx_yp = idx + nx;
        int idx_zm = idx - nx * ny;
        int idx_zp = idx + nx * ny;

        double laplacian = (phi[idx_xp] - 2.0 * phi[idx] + phi[idx_xm]) / (dx * dx) +
                           (phi[idx_yp] - 2.0 * phi[idx] + phi[idx_ym]) / (dy * dy) +
                           (phi[idx_zp] - 2.0 * phi[idx] + phi[idx_zm]) / (dz * dz);

        double res = laplacian - f[idx];
        residual_array[idx] = res * res;
    }
    else if (i < nx && j < ny && k < nz)
    {
        int idx = i + j * nx + k * nx * ny;
        residual_array[idx] = 0.0;
    }
}

__global__ void computeError(double *phi, double *phi_exact, double *error_array,
                             int nx, int ny, int nz_local, int z_start)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (i < nx && j < ny && k < nz_local)
    {
        int global_k = z_start + k - 1;
        int idx = i + j * nx + k * nx * ny;

        if (global_k >= 0 && global_k < NZ &&
            i >= 0 && i < nx &&
            j >= 0 && j < ny)
        {
            double diff = phi[idx] - phi_exact[idx];
            error_array[idx] = diff * diff;
        }
        else
        {
            error_array[idx] = 0.0;
        }
    }
}

#endif // TASK2_CUH