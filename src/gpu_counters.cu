#include <iostream>
#include <time.h>

#include "cuda_utils.h"

struct Stats
{
    unsigned int even;
};

__global__ void test_kernel(int* A, int* B, Stats* stats)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int res = A[tid] + (int)tid;

    if (res%2 == 0)
	atomicAdd(&(stats->even), 1);

    B[tid] = res;
}

int get_random_int(int min, int max)
{
    return min + (rand() % (int)(max - min + 1));
}

void print_array(int* ar, unsigned int n)
{
    for (unsigned int i = 0; i < n; ++i)
	std::cout << ar[i] << " ";
    std::cout << std::endl;
}

void print_stats(Stats* s)
{
    std::cout << "even: " << s->even << std::endl;
}

int main()
{
    // vector size
    const unsigned int N = 10;

    // device vectors
    int *d_A, *d_B;
    Stats *d_stats;

    // host vectors
    int *h_A, *h_B;
    Stats *h_stats;

    // allocate device memory
    CUDA_SAFE_CALL(cudaMalloc(&d_A, N * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_B, N * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_stats, sizeof(Stats)));

    // allocate host memory
    h_A = new int[N];
    h_B = new int[N];
    h_stats = new Stats;

    // initialize host data
    srand(time(NULL));
    for (unsigned int i = 0; i < N; ++i)
    {
        h_A[i] = get_random_int(0,10);
        h_B[i] = 0;
    }
    memset(h_stats, 0, sizeof(Stats));

    // copy data to the device
    CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_stats, h_stats, sizeof(Stats), cudaMemcpyHostToDevice));

    // launch kernel
    dim3 grid_size, block_size;
    grid_size.x = N;
    test_kernel<<<grid_size, block_size>>>(d_A, d_B, d_stats);

    // copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_B, d_B, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_stats, d_stats, sizeof(Stats), cudaMemcpyDeviceToHost));

    print_array(h_B, N);
    print_stats(h_stats);

    // free device memory
    CUDA_SAFE_CALL(cudaFree(d_A));
    CUDA_SAFE_CALL(cudaFree(d_B));
    CUDA_SAFE_CALL(cudaFree(d_stats));

    // free host memory
    delete [] h_A;
    delete [] h_B;
    delete h_stats;
}
