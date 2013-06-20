#include <iostream>
#include <assert.h>

#include "cuda_utils.h"

__device__ unsigned int __isLocal(const void *ptr)
{
  unsigned int ret;
  asm volatile ("{ \n\t"
                "    .reg .pred p; \n\t"
                "    isspacep.local p, %1; \n\t"
                "    selp.u32 %0, 1, 0, p;  \n\t"
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
                "} \n\t" : "=r"(ret) : "l"(ptr));
#else
                "} \n\t" : "=r"(ret) : "r"(ptr));
#endif

  return ret;
}

__device__ void dummy_func(int& n)
{
    // Each thread modifies a local register, however this would not work
    // if n were stored in global or shared memory (WAW hazard).
    assert(__isLocal(&n));

    n = threadIdx.x;
}

__global__ void test_kernel(int* A)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int res = -1;
    dummy_func(res);

    A[tid] = res;
}

void print_array(int* ar, unsigned int n)
{
    for (unsigned int i = 0; i < n; ++i)
	std::cout << ar[i] << " ";
    std::cout << std::endl;
}

int main()
{
    // vector size
    const unsigned int N = 10;

    // device vector
    int *d_A;

    // host vector
    int *h_A;

    // allocate device memory
    CUDA_SAFE_CALL(cudaMalloc(&d_A, N * sizeof(int)));

    // allocate host memory
    h_A = new int[N];

    // initialize host data
    memset(h_A, 0, N * sizeof(int));

    // copy data to the device
    CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice));

    // launch kernel
    dim3 grid_size, block_size;
    block_size.x = N;
    test_kernel<<<grid_size, block_size>>>(d_A);

    // copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost));

    print_array(h_A, N);

    // free device memory
    CUDA_SAFE_CALL(cudaFree(d_A));

    // free host memory
    delete [] h_A;
}
