#include <iostream>
#include <assert.h>

#include "cuda_utils.h"

template <typename T>
__global__ void test_kernel(T* A)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    A[tid] = (T)tid;

    // This works
    assert (A[tid] >= 0);

    // This fails
    assert (A[tid]
            >= 0);
}


template <typename T>
void print_array(T* ar, uint N)
{
    for (uint i = 0; i < N; ++i)
        std::cout << ar[i] << " ";
    std::cout << std::endl;
}


template <typename T>
void run_kernel (uint N)
{
    // host array
    T* h_A;

    // device array
    T* d_A;

    // allocate device memory
    CUDA_SAFE_CALL (cudaMalloc (&d_A, N * sizeof (T)));

    // allocate host memory
    h_A = (T*) malloc (N * sizeof (T));

    // initialize host data
    memset (h_A, 0, N * sizeof (T));

    // copy data to the device
    CUDA_SAFE_CALL (cudaMemcpy (d_A, h_A, N * sizeof (T), cudaMemcpyHostToDevice));

    // launch kernel
    dim3 grid_size, block_size;
    block_size.x = N;
    test_kernel<<<grid_size, block_size>>> (d_A);

    // copy result back to host
    CUDA_SAFE_CALL (cudaMemcpy (h_A, d_A, N * sizeof (T), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL (cudaDeviceSynchronize());

    print_array (h_A, N);

    // free device memory
    CUDA_SAFE_CALL (cudaFree (d_A));

    // free host memory
    free (h_A);
}


int main()
{
    typedef int type_t;

    // vector size
    const uint N = 10;

    // run kernel
    run_kernel<type_t> (N);
}
