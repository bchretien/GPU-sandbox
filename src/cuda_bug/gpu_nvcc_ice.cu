#include <iostream>
#include <stdio.h>

#include "cuda_utils.h"

template <typename T, unsigned int N>
__device__ inline void right_func (T (&ar)[N])
{
    ar[threadIdx.x] = 1.0;
}

template <typename T, unsigned int N>
__device__ inline void wrong_func (T (ar&)[N])
{
    ar[threadIdx.x] = 1.0;
}

template <typename T, unsigned int N>
__global__ void test_kernel (T *vec)
{
    __shared__ T shared_vec[10];

    right_func<T,10> (shared_vec);
    wrong_func<T,10> (shared_vec);
}

int main()
{
    typedef float type_t;
    const unsigned int N = 128;

    type_t h_vec[N];
    type_t* d_vec;

    CUDA_SAFE_CALL (cudaMalloc (&d_vec, N * sizeof(type_t)));
    test_kernel<type_t, N><<<1, N>>> (d_vec);
    CUDA_SAFE_CALL (cudaFree(d_vec));
}
