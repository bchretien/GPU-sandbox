/// Texture memory vs Global memory vs Constant memory test
/// See: http://stackoverflow.com/questions/14398416/convenience-of-2d-cuda-texture-memory-against-global-memory

#include <cuda.h>
#include <stdio.h>
#include <algorithm>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define ARRAY_SIZE_X 128
#define ARRAY_SIZE_Y 128

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

template <typename T>
__global__ void initMatrix(T *matrix, unsigned int width, unsigned int height, T val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < width * height; i += gridDim.x * blockDim.x)
    {
        matrix[i] = val;
    }
}

// Reference to 2D texture
texture<float,2> data_texture;

// Constant memory data
__constant__ float data_constant[ARRAY_SIZE_X * ARRAY_SIZE_Y];

/*********************************/
/* 2D TEXTURE MEMORY TEST KERNEL */
/*********************************/
__global__ void Kernel_BasicTest_Texture(float* dev_result, int N1, int N2)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < N1 && j < N2)
    {
        dev_result[j*blockDim.x*gridDim.x + i] *= tex2D(data_texture, i, j);
    }
}

__global__ void Kernel_ComplexTest_Texture(float* dev_result, int N1, int N2)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    float datum, accumulator = 0.;

    int size_x = 5;
    int size_y = 5;

    if((i < N1-size_x) && (j < N2-size_y))
    {
        for (int k = 0; k < size_x; k++)
            for (int l = 0; l < size_y; l++)
            {
                datum = tex2D(data_texture, i+k, j+l);
                accumulator = accumulator + datum;
            }
        dev_result[j*blockDim.x*gridDim.x+i] = accumulator;
    }
}

/**********************************/
/* 2D CONSTANT MEMORY TEST KERNEL */
/**********************************/
__global__ void Kernel_BasicTest_Constant(float* dev_result, int N1, int N2)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < N1 && j < N2)
    {
        dev_result[j*blockDim.x*gridDim.x + i] *= data_constant[j*blockDim.x*gridDim.x + i];
    }
}

__global__ void Kernel_ComplexTest_Constant(float* dev_result, int N1, int N2)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    float accumulator = 0.;

    int size_x = 5;
    int size_y = 5;

    if((i < N1-size_x) && (j < N2-size_y))
    {
        for (int k = 0; k < size_x; k++)
            for (int l = 0; l < size_y; l++)
            {
                accumulator = accumulator + data_constant[(j+l)*blockDim.x*gridDim.x + (i+k)];
            }
        dev_result[j*blockDim.x*gridDim.x+i] = accumulator;
    }
}

/********************************/
/* 2D GLOBAL MEMORY TEST KERNEL */
/********************************/
__global__ void Kernel_BasicTest_Global(float* dev_result, float* data_d, int N1, int N2)
{    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < N1 && j < N2)
    {
        dev_result[j*blockDim.x*gridDim.x+i] *= data_d[j*blockDim.x*gridDim.x + i];
    }
}

__global__ void Kernel_ComplexTest_Global(float* dev_result, float* data_d, int N1, int N2)
{    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    float accumulator = 0.;

    int size_x = 5;
    int size_y = 5;

    if((i < N1-size_x) && (j < N2-size_y))
    {
        for (int k = 0; k < size_x; k++)
            for (int l = 0; l < size_y; l++)
            {
                accumulator = accumulator + data_d[(j+l)*blockDim.x*gridDim.x + (i+k)];
            }
        dev_result[j*blockDim.x*gridDim.x + i] = accumulator;
    }
}

/**************************/
/* TEXTURE MEMORY VERSION */
/**************************/
void Test_Texture(float* h_data, float* d_result1, float* h_result1, float* d_result2, float* h_result2, int N1, int N2)
{
    size_t pitch; 
    float* d_data;

    // Copy data to texture memory
    CUDA_SAFE_CALL(cudaMallocPitch((void**)&d_data, &pitch, N1 * sizeof(float), N2));
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    CUDA_SAFE_CALL(cudaBindTexture2D(0, &data_texture, d_data, &desc, N1, N2, pitch));
    CUDA_SAFE_CALL(cudaMemcpy2D(d_data, pitch, h_data,
                                sizeof(float) * N1, sizeof(float) * N1, N2,
                                cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
    dim3 dimGrid(N1/BLOCK_SIZE_X + (N1%BLOCK_SIZE_X == 0 ? 0:1),
                 N2/BLOCK_SIZE_X + (N2%BLOCK_SIZE_X == 0 ? 0:1));

    // Run the basic kernel
    Kernel_BasicTest_Texture<<<dimGrid,dimBlock>>>(d_result1, N1, N2);
    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_result1, d_result1, N1 * N2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Run the complex kernel
    Kernel_ComplexTest_Texture<<<dimGrid,dimBlock>>>(d_result2, N1, N2);
    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_result2, d_result2, N1 * N2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_SAFE_CALL(cudaFree(d_data));
}

/***************************/
/* CONSTANT MEMORY VERSION */
/***************************/
void Test_Constant(float* h_data,
                   float* d_result1, float* h_result1,
                   float* d_result2, float* h_result2,
                   int N1, int N2)
{
    // Copy data to constant memory
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(data_constant, h_data, N1 * N2 * sizeof(float)));

    dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
    dim3 dimGrid(N1/BLOCK_SIZE_X + (N1%BLOCK_SIZE_X == 0 ? 0:1),
                 N2/BLOCK_SIZE_X + (N2%BLOCK_SIZE_X == 0 ? 0:1));

    // Run the basic kernel
    Kernel_BasicTest_Constant<<<dimGrid,dimBlock>>>(d_result1, N1, N2);
    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_result1, d_result1, N1 * N2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Run the complex kernel
    Kernel_ComplexTest_Constant<<<dimGrid,dimBlock>>>(d_result2, N1, N2);
    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_result2, d_result2, N1 * N2 * sizeof(float), cudaMemcpyDeviceToHost));
}

/*************************/
/* GLOBAL MEMORY VERSION */
/*************************/
void Test_Global(float* h_data,
                 float* d_result1, float* h_result1,
                 float* d_result2, float* h_result2,
                 int N1, int N2)
{
    float* d_data;

    // Allocate global memory
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, N1 * N2 * sizeof(float)));
    // Copy host data to global memory
    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, N1 * N2 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
    dim3 dimGrid(N1/BLOCK_SIZE_X + (N1%BLOCK_SIZE_X == 0 ? 0:1),
                 N2/BLOCK_SIZE_X + (N2%BLOCK_SIZE_X == 0 ? 0:1));

    // Run the basic kernel
    Kernel_BasicTest_Global<<<dimGrid,dimBlock>>>(d_result1, d_data, N1, N2);
    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_result1, d_result1, N1 * N2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Run the complex kernel
    Kernel_ComplexTest_Global<<<dimGrid,dimBlock>>>(d_result2, d_data, N1, N2);
    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_result2, d_result2, N1 * N2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_SAFE_CALL(cudaFree(d_data));
}


/******************/
/* COMPARE ARRAYS */
/******************/
bool are_equal(float* h_A, float* h_B, float* h_C, int N1, int N2)
{
    for (int i = 0; i < N1 * N2; ++i)
        if (h_A[i] != h_B[i] || h_A[i] != h_C[i]) return false;

    return true;
}

int main()
{
    unsigned int n1 = ARRAY_SIZE_X;
    unsigned int n2 = ARRAY_SIZE_Y;

    float *h_data;
    float *d_A1, *d_B1, *d_C1, *h_resA1, *h_resB1, *h_resC1;
    float *d_A2, *d_B2, *d_C2, *h_resA2, *h_resB2, *h_resC2;

    // Allocate host memory
    h_data  = (float*) malloc(n1 * n2 * sizeof(float));
    h_resA1 = (float*) malloc(n1 * n2 * sizeof(float));
    h_resA2 = (float*) malloc(n1 * n2 * sizeof(float));
    h_resB1 = (float*) malloc(n1 * n2 * sizeof(float));
    h_resB2 = (float*) malloc(n1 * n2 * sizeof(float));
    h_resC1 = (float*) malloc(n1 * n2 * sizeof(float));
    h_resC2 = (float*) malloc(n1 * n2 * sizeof(float));

    // Allocate device memory
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_A1, n1 * n2 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_A2, n1 * n2 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_B1, n1 * n2 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_B2, n1 * n2 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_C1, n1 * n2 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_C2, n1 * n2 * sizeof(float)));

    // Set device memory to 1 (basic test) or 0 (complex test)
    dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
    dim3 dimGrid(n1/BLOCK_SIZE_X + (n1%BLOCK_SIZE_X == 0 ? 0:1),
                 n1/BLOCK_SIZE_X + (n2%BLOCK_SIZE_X == 0 ? 0:1));
    initMatrix<float><<<dimGrid, dimBlock>>>(d_A1, n1, n2, 1.0);
    initMatrix<float><<<dimGrid, dimBlock>>>(d_B1, n1, n2, 1.0);
    initMatrix<float><<<dimGrid, dimBlock>>>(d_C1, n1, n2, 1.0);
    CUDA_SAFE_CALL(cudaMemset(d_A2, 0.0, n1 * n2 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(d_B2, 0.0, n1 * n2 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(d_C2, 0.0, n1 * n2 * sizeof(float)));

    // Set test data on host
    std::fill(h_data, h_data + (n1*n2), 2.0 ) ;

    // Launch the 3 tests: texture memory, constant memory and global memory
    Test_Texture(h_data, d_B1, h_resB1, d_B2, h_resB2, n1, n2);
    Test_Constant(h_data, d_C1, h_resC1, d_C2, h_resC2, n1, n2);
    Test_Global(h_data, d_A1, h_resA1, d_A2, h_resA2, n1, n2);

    // Compare the results that should be equal
    if (!are_equal(h_resA1, h_resB1, h_resC1, n1, n2)
     || !are_equal(h_resA2, h_resB2, h_resC2, n1, n2))
    {
        printf("Error: texture memory, constant memory and global memory outputs are inconsistent.\n");
    }

    // Free device memory
    CUDA_SAFE_CALL(cudaFree(d_A1));
    CUDA_SAFE_CALL(cudaFree(d_B1));
    CUDA_SAFE_CALL(cudaFree(d_C1));
    CUDA_SAFE_CALL(cudaFree(d_A2));
    CUDA_SAFE_CALL(cudaFree(d_B2));
    CUDA_SAFE_CALL(cudaFree(d_C2));

    // Free host memory
    free(h_data);
    free(h_resA1);
    free(h_resA2);
    free(h_resB1);
    free(h_resB2);
    free(h_resC1);
    free(h_resC2);
}
