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

__global__ void jacobiIterationStream(double *phi, double *phi_new, double *f, 
                                      int nx, int ny, int nz, 
                                      double dx, double dy, double dz, 
                                      double *diff_array, int z_start, int z_end)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index
    int k = blockIdx.z + z_start;                 // z index (adjusted for stream)

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1 && k >= z_start && k < z_end)
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
    else if (i < nx && j < ny && k >= z_start && k < z_end)
    {
        double x = i * dx;
        double y = j * dy;
        double z = k * dz;
        int idx = i + j * nx + k * nx * ny;

        phi_new[idx] = sin(N_PARAM * PI * x) * cos(M_PARAM * PI * y) * sin(K_PARAM * PI * z);
        diff_array[idx] = 0.0;
    }
}

__global__ void jacobiIterationShared(double *phi, double *phi_new, double *f, 
                                      int nx, int ny, int nz, 
                                      double dx, double dy, double dz, 
                                      double *diff_array)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index
    int k = blockIdx.z;                            // z index

    // To use Shared Memory, we need to allocate extra space for the points in the current block and the surrounding halo points
    // Shared Memory size: (blockDim.x + 2) * (blockDim.y + 2)
    extern __shared__ double sdata[]; 

    // sWidth and sHeight are the effective dimensions of the shared memory 2D array
    const int sWidth = blockDim.x + 2;
    const int sHeight = blockDim.y + 2;

    // Coordinates of the current thread in shared memory (adding 1 to leave space for halo)
    int s_i = threadIdx.x + 1;
    int s_j = threadIdx.y + 1;

    // Index in the global data array
    int idx = i + j * nx + k * nx * ny;
    
    // When accessing global memory, pay attention to boundary conditions, if out of range set to 0 (or appropriate boundary conditions)
    double val = 0.0;
    if (i < nx && j < ny && k < nz)
        val = phi[idx];

    // Write own point to shared memory
    sdata[s_j * sWidth + s_i] = val;

    if (threadIdx.x == 0 && i > 0) {
        sdata[s_j * sWidth + (s_i - 1)] = (j < ny && k < nz) ? phi[idx - 1] : 0.0;
    }
    if (threadIdx.x == blockDim.x - 1 && i < nx - 1) {
        sdata[s_j * sWidth + (s_i + 1)] = (j < ny && k < nz) ? phi[idx + 1] : 0.0;
    }
    if (threadIdx.y == 0 && j > 0) {
        sdata[(s_j - 1) * sWidth + s_i] = (i < nx && k < nz) ? phi[idx - nx] : 0.0;
    }
    if (threadIdx.y == blockDim.y - 1 && j < ny - 1) {
        sdata[(s_j + 1) * sWidth + s_i] = (i < nx && k < nz) ? phi[idx + nx] : 0.0;
    }

    // Handle the four corner halo data (only load if the corresponding thread is at the block corner)
    if (threadIdx.x == 0 && threadIdx.y == 0 && i > 0 && j > 0) {
        sdata[(s_j - 1)*sWidth + (s_i - 1)] = (k < nz) ? phi[idx - nx - 1] : 0.0;
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && i < nx-1 && j > 0) {
        sdata[(s_j - 1)*sWidth + (s_i + 1)] = (k < nz) ? phi[idx - nx + 1] : 0.0;
    }
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && i > 0 && j < ny-1) {
        sdata[(s_j + 1)*sWidth + (s_i - 1)] = (k < nz) ? phi[idx + nx - 1] : 0.0;
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && i < nx-1 && j < ny-1) {
        sdata[(s_j + 1)*sWidth + (s_i + 1)] = (k < nz) ? phi[idx + nx + 1] : 0.0;
    }

    __syncthreads();

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
    {
        // Get neighboring values in x and y directions from shared memory
        double phi_xm = sdata[s_j * sWidth + (s_i - 1)];
        double phi_xp = sdata[s_j * sWidth + (s_i + 1)];
        double phi_ym = sdata[(s_j - 1) * sWidth + s_i];
        double phi_yp = sdata[(s_j + 1) * sWidth + s_i];

        // Neighboring values in z direction still need to be read from global memory, as they are from different z layers
        int idx_xm = idx - 1;
        int idx_xp = idx + 1;
        int idx_ym = idx - nx;
        int idx_yp = idx + nx;
        int idx_zm = idx - nx * ny;
        int idx_zp = idx + nx * ny;

        double phi_zm = phi[idx_zm];
        double phi_zp = phi[idx_zp];

        double phi_old = sdata[s_j * sWidth + s_i]; // Original value already in shared memory

        double denom = 2.0 / (dx * dx) + 2.0 / (dy * dy) + 2.0 / (dz * dz);
        double numerator = (phi_xm + phi_xp) / (dx * dx) +
                           (phi_ym + phi_yp) / (dy * dy) +
                           (phi_zm + phi_zp) / (dz * dz) - f[idx];

        double phi_new_val = numerator / denom;

        phi_new[idx] = phi_new_val;

        double diff_local = phi_new_val - phi_old;
        diff_array[idx] = diff_local * diff_local;
    }
    else if (i < nx && j < ny && k < nz)
    {
        // Boundary conditions
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
