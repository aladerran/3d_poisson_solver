// task1.cuh
#ifndef TASK1_CUH
#define TASK1_CUH

#pragma once

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

#include <math.h>

#define PI 3.14159265358979323846
#define N_PARAM 2
#define M_PARAM 2
#define K_PARAM 2

__global__ void initializePhi(double *phi, double *phi_new, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index
    int k = blockIdx.z;                            // z index

    if (i < nx && j < ny && k < nz)
    {
        int idx = i + j * nx + k * nx * ny;
        phi[idx] = 0.0;
        phi_new[idx] = 0.0;
    }
}

__global__ void initializeF(double *f, int nx, int ny, int nz, double dx, double dy, double dz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index
    int k = blockIdx.z;                            // z index

    if (i < nx && j < ny && k < nz)
    {
        double x = i * dx;
        double y = j * dy;
        double z = k * dz;

        int idx = i + j * nx + k * nx * ny;

        double phi_val = sin(N_PARAM * PI * x) * cos(M_PARAM * PI * y) * sin(K_PARAM * PI * z);
        f[idx] = - (N_PARAM * N_PARAM + M_PARAM * M_PARAM + K_PARAM * K_PARAM) * PI * PI * phi_val;
    }
}

__global__ void computeExactSolution(double *phi_exact, int nx, int ny, int nz, double dx, double dy, double dz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index
    int k = blockIdx.z;                            // z index

    if (i < nx && j < ny && k < nz)
    {
        double x = i * dx;
        double y = j * dy;
        double z = k * dz;

        int idx = i + j * nx + k * nx * ny;

        phi_exact[idx] = sin(N_PARAM * PI * x) * cos(M_PARAM * PI * y) * sin(K_PARAM * PI * z);
    }
}

__global__ void jacobiIteration(double *phi, double *phi_new, double *f, int nx, int ny, int nz, double dx, double dy, double dz, double *diff_array)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index
    int k = blockIdx.z;                            // z index

    if (i >= 1 && i < nx - 1 && j >=1 && j < ny - 1 && k >=1 && k < nz -1)
    {
        int idx = i + j * nx + k * nx * ny;

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
    else if (i < nx && j < ny && k < nz)
    {
        // Boundary conditions: phi = exact solution
        double x = i * dx;
        double y = j * dy;
        double z = k * dz;
        int idx = i + j * nx + k * nx * ny;

        phi_new[idx] = sin(N_PARAM * PI * x) * cos(M_PARAM * PI * y) * sin(K_PARAM * PI * z);
        diff_array[idx] = 0.0;
    }
}

__global__ void computeResidual(double *phi, double *f, double *residual_array, int nx, int ny, int nz, double dx, double dy, double dz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index
    int k = blockIdx.z;                            // z index

    if (i >= 1 && i < nx - 1 && j >=1 && j < ny - 1 && k >=1 && k < nz -1)
    {
        int idx = i + j * nx + k * nx * ny;

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

__global__ void computeError(double *phi, double *phi_exact, double *error_array, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index
    int k = blockIdx.z;                            // z index

    if (i < nx && j < ny && k < nz)
    {
        int idx = i + j * nx + k * nx * ny;

        double diff = phi[idx] - phi_exact[idx];
        error_array[idx] = diff * diff;
    }
}

#endif // TASK1_CUH
