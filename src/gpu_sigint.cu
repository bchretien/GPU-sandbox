#include <iostream>
#include <signal.h>
#include "cuda_utils.h"

void cuda_sig_handler(int sig)
{
    bool is_handled = false;

    switch (sig)
    {
    case SIGINT:
        std::cerr << "\nCaught SIGINT. ";
        is_handled = true;
        break;

    case SIGTERM:
        std::cerr << "\nCaught SIGTERM. ";
        is_handled = true;
        break;

    default:
        std::cerr << "\nUnhandled signal. Doing nothing.";
        is_handled = false;
        return;
    }

    if (is_handled)
    {
        // Reset GPU
        std::cerr << "Resetting device(s)." << std::endl;
        int devCount;
        cudaGetDeviceCount(&devCount);
        for (int i = 0; i < devCount; ++i)
        {
            cudaSetDevice(i);
            cudaDeviceReset();
        }
        exit(sig);
    }
}


__global__ void dummy_kernel(int* A)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    A[tid] = 42;
}


int main()
{
    // Catch signals
    if (signal(SIGINT, cuda_sig_handler) == SIG_ERR)
        std::cerr << "Cannot catch SIGINT" << std::endl;
    if (signal(SIGTERM, cuda_sig_handler) == SIG_ERR)
        std::cerr << "Cannot catch SIGTERM" << std::endl;

    // Vector size: 2^24 elements
    const unsigned int N = 1 << 24;

    // Device vector
    int *d_A;

    // Host vector
    int *h_A;

    // Allocate device memory
    CUDA_SAFE_CALL(cudaMalloc(&d_A, N * sizeof(int)));

    // Allocate host memory
    h_A = new int[N];

    // Initialize host data
    memset(h_A, 0, N * sizeof(int));

    // Copy data to the device
    CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice));

    dim3 grid_size, block_size;
    block_size.x = 512;
    grid_size.x = N/block_size.x;

    // Long loop that gives time for the user to hit ctrl+C
    for (unsigned int i = 0; i < 1e9; ++i)
    {
        dummy_kernel<<<grid_size, block_size>>>(d_A);
    }

    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_SAFE_CALL(cudaFree(d_A));

    // Free host memory
    delete [] h_A;
}
