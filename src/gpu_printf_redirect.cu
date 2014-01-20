// See: http://stackoverflow.com/questions/21238303/redirecting-cuda-printf-to-a-c-stream

#include <cuda.h>
#include <stdio.h>
#include <unistd.h> // dup

#include <iostream>
#include <sstream> // stringstream
#include <fstream> // ofstream

#include "cuda_utils.h"

char* output_file = "printf_redirect.log";

__global__ void printf_redirect(int* src, int* res)
{
    res[threadIdx.x] = threadIdx.x;
    printf("  %i: Hello World!\n", res[threadIdx.x]);
}

int main()
{
    using namespace std;

    const uint N = 2;

    // Note: dummy arrays are not actually used, but this should prevent the
    //       compiler from discarding the printf in the kernel.

    int *d_A, *d_B, *h_A, *h_B;
    size_t size = N * sizeof (int);
    CUDA_SAFE_CALL (cudaMalloc (&d_A, size));
    CUDA_SAFE_CALL (cudaMalloc (&d_B, size));
    h_A = (int*) malloc (size);
    h_B = (int*) malloc (size);
    CUDA_SAFE_CALL (cudaMemcpy (d_A, h_A, size, cudaMemcpyHostToDevice));

    std::cout << "std::cout - start" << std::endl;
    printf ("stdout - start\n");

    /// REGULAR PRINT
    // Print to regular stdout
    std::cout << "Output to stdout:" << std::endl;
    printf_redirect<<<1,1>>> (d_A, d_B);
    CUDA_SAFE_CALL (cudaDeviceSynchronize ());

    /// REDIRECTION TO STRINGSTREAM
    std::stringstream ss;
    // Redirect std::cout to a stringstream
    std::streambuf* backup_cout = std::cout.rdbuf ();
    std::cout.rdbuf (ss.rdbuf ());
    // Redirect stdout to a buffer
    char buf[1024] = "";
    int backup_stdout = dup (fileno (stdout));
    freopen ("/dev/null", "w", stdout);
    setbuf (stdout, buf);

    std::cout << "Redirected output:" << std::endl;
    printf_redirect<<<1,N>>> (d_A, d_B);
    CUDA_SAFE_CALL (cudaDeviceSynchronize ());

    // Add CUDA buffer to a stringstream
    ss << buf;

    // Write stringstream to file
    std::ofstream outFile;
    outFile.open (output_file);
    outFile << ss.str ();
    outFile.close ();

    /// RESET REDIRECTION
    // Redirect back to initial stdout
    fflush (stdout);
    setbuf (stdout, NULL);
    fclose (stdout);
    FILE *fp = fdopen (backup_stdout, "w");
    fclose (stdout);
    *stdout = *fp;
    // Redirect back to initial std::cout
    std::cout.rdbuf (backup_cout);

    std::cout << "std::cout - end" << std::endl;
    printf ("stdout - end\n");

    CUDA_SAFE_CALL (cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL (cudaFree(d_A));
    CUDA_SAFE_CALL (cudaFree(d_B));
    free (h_A);
    free (h_B);
}
