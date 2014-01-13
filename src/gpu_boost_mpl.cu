// Code used to answer: http://stackoverflow.com/questions/21093263/is-it-possible-to-use-stdconditional-in-a-cuda-kernel
#include <cuda.h>
#include <stdio.h>

#include <boost/type_traits/conditional.hpp>
#include <cuComplex.h>

#include "cuda_utils.h"

#include <iostream>

template <typename fType>
__global__ void DummyKernel(fType* src, fType* res) {
    typedef typename boost::conditional<sizeof(fType) == sizeof(double),cuDoubleComplex, cuFloatComplex>::type cfType;

    printf("  size = %u bytes\n", sizeof (cfType));
}

int main()
{
    const uint N = 1;

    // Note: dummy arrays are not actually used, but this should prevent the
    //       compiler from discarding the printf in the kernel.

    float *d_A, *d_B, *h_A, *h_B;
    size_t size = N * sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc(&d_A, size));
    CUDA_SAFE_CALL(cudaMalloc(&d_B, size));
    h_A = (float*) malloc (size);
    h_B = (float*) malloc (size);
    CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    std::cout << "Float complex:" << std::endl;
    DummyKernel<float><<<1,1>>>(d_A, d_B);

    CUDA_SAFE_CALL(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));

    double *d_C, *d_D, *h_C, *h_D;
    size = N * sizeof(double);
    CUDA_SAFE_CALL(cudaMalloc(&d_C, size));
    CUDA_SAFE_CALL(cudaMalloc(&d_D, size));
    h_C = (double*) malloc (size);
    h_D = (double*) malloc (size);
    CUDA_SAFE_CALL(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

    std::cout << "Double complex:" << std::endl;
    DummyKernel<double><<<1,1>>>(d_C, d_D);

    CUDA_SAFE_CALL(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_A));
    CUDA_SAFE_CALL(cudaFree(d_B));
    CUDA_SAFE_CALL(cudaFree(d_C));
    CUDA_SAFE_CALL(cudaFree(d_D));
    free (h_A);
    free (h_B);
    free (h_C);
    free (h_D);
}
