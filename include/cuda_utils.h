#include <cuda.h>
#include <stdio.h>

#define CUDA_CHECK_ERROR()  __cuda_check_errors(__FILE__, __LINE__)
#define CUDA_SAFE_CALL(err) __cuda_safe_call(err, __FILE__, __LINE__)

inline void __cuda_check_errors(const char *filename, const int line_number)
{
    cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        printf("CUDA error %i at %s:%i: %s\n",
               err, filename, line_number, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __cuda_safe_call(cudaError err, const char *filename, const int line_number)
{
    if (err != cudaSuccess)
    {
        printf("CUDA error %i at %s:%i: %s\n",
               err, filename, line_number, cudaGetErrorString(err));
        exit(-1);
    }
}
