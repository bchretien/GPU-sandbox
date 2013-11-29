// vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

#include <iostream>
#include <assert.h>

#include <thrust/host_vector.h>

#include "cuda_utils.h"

template <typename T>
struct ArrayList
{
    T*   ar;
};


template <typename T>
struct VectorList
{
    std::vector<T>  ar;
};


template <typename T>
__global__ void set_kernel (ArrayList<T>* A, uint M)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (uint i = 0; i < M; ++i)
        A[tid].ar[i] += tid;
}


template <typename T>
void print_arrays (const ArrayList<T>* v, uint N, uint M)
{
    for (uint i = 0; i < N; ++i)
    {
        std::cout << "i = " << i << ": ";
        for (uint j = 0; j < M; ++j)
            std::cout << v[i].ar[j] << " ";
        std::cout << std::endl;
    }
}


template <typename T>
void print_vectors (const std::vector<VectorList<T> >& v, uint N, uint M)
{
    for (uint i = 0; i < N; ++i)
    {
        std::cout << "i = " << i << ": ";
        for (uint j = 0; j < M; ++j)
            std::cout << v[i].ar[j] << " ";
        std::cout << std::endl;
    }
}


template <typename T>
void run_kernel (ArrayList<T>*& h_A,
                 ArrayList<T>*& d_A,
                 uint N, uint M)
{
    // Allocate outer memory
    size_t outer_size = N * sizeof (ArrayList<T>);
    CUDA_SAFE_CALL (cudaHostAlloc( (void**)&h_A, outer_size, cudaHostAllocMapped));
    CUDA_SAFE_CALL (cudaHostGetDevicePointer(&d_A, h_A, 0));

    size_t inner_size = M * sizeof (T);
    for (size_t i = 0; i < N; ++i)
    {
        // Allocate inner memory
        CUDA_SAFE_CALL (cudaHostAlloc ((void**)&h_A[i].ar, inner_size, cudaHostAllocMapped));
        CUDA_SAFE_CALL (cudaHostGetDevicePointer(&d_A[i].ar, h_A[i].ar, 0));

        // initialize host data
        memset (h_A[i].ar, 0, inner_size);
        h_A[i].ar[0] = 1;
    }
    CUDA_SAFE_CALL (cudaDeviceSynchronize());

    std::cout << "Before kernel call:" << std::endl;
    print_arrays (h_A, N, M);

    // Launch kernel
    dim3 grid_size, block_size;
    block_size.x = N;
    set_kernel<<<grid_size, block_size>>> (d_A, M);
    CUDA_CHECK_ERROR ();

    // Copy result back to host automatically
    CUDA_SAFE_CALL (cudaDeviceSynchronize());

    std::cout << "After kernel call:" << std::endl;
    print_arrays (h_A, N, M);

    // Test map to STL vector
    std::cout << "After copy to STL vector:" << std::endl;
    std::vector<VectorList<T> > v;
    v.resize (N);
    for (size_t i = 0; i < N; ++i)
    {
        v[i].ar.resize (M);
        std::copy (&h_A[i].ar[0], &h_A[i].ar[M], v[i].ar.begin ());
    }
    print_vectors (v, N, M);

    // Free host+device memory
    for (size_t i = 0; i < N; ++i)
        CUDA_SAFE_CALL (cudaFreeHost (h_A[i].ar));
    CUDA_SAFE_CALL (cudaFreeHost (h_A));
}


int main()
{
    typedef int type_t;

    // vector list size
    const uint N = 4;

    // vector size
    const uint M = 4;

    // host array
    ArrayList<type_t>* h_A;

    // device array
    ArrayList<type_t>* d_A;

    // run kernel
    run_kernel<type_t> (h_A, d_A, N, M);
}
