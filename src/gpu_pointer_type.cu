#include <iostream>
#include <stdio.h>

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

__device__ unsigned int __isShared(const void *ptr)
{
  unsigned int ret;
  asm volatile ("{ \n\t"
                "    .reg .pred p; \n\t"
                "    isspacep.shared p, %1; \n\t"
                "    selp.u32 %0, 1, 0, p;  \n\t"
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
                "} \n\t" : "=r"(ret) : "l"(ptr));
#else
                "} \n\t" : "=r"(ret) : "r"(ptr));
#endif

  return ret;
}

__device__ void analyze_pointer(const void *ptr)
{
    printf("\t* is local:  %u\n", __isLocal(ptr));
    printf("\t* is global: %u\n", __isGlobal(ptr));
    printf("\t* is shared: %u\n", __isShared(ptr));
}

template <typename T, unsigned int N>
__global__ void test_kernel(T *vec)
{
    // Shared array
    __shared__ T shared_vec[10];

    // Register array
    T reg[10];

    // Thread id in the global array
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    vec[tid] = 1;

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("Register array:\n");
        analyze_pointer(&reg);

        printf("\nGlobal array:\n");
        analyze_pointer(vec);

        printf("\nShared array:\n");
        analyze_pointer(&shared_vec);
    }
}

int main()
{
    typedef float type_t;
    const unsigned int N = 128;

    type_t h_vec[N];
    type_t* d_vec;
    
    //for (unsigned int i = 0; i < N; ++i)
    //    h_vec[i] = 0;

    CUDA_SAFE_CALL(cudaMalloc(&d_vec, N * sizeof(type_t)));
    //CUDA_SAFE_CALL(cudaMemcpy(d_vec, h_vec, N * sizeof(type_t), cudaMemcpyHostToDevice));

    test_kernel<type_t, N><<<1, N>>>(d_vec);

    //CUDA_SAFE_CALL(cudaMemcpy(h_vec, d_vec, N * sizeof(type_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_vec));
}
