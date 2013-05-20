#include <string.h>
#include <iostream>
#include <stdio.h>

#define INIT_A 0x67452301
#define INIT_B 0xefcdab89
#define INIT_C 0x98badcfe
#define INIT_D 0x10325476

#define SQRT_2 0x5a827999
#define SQRT_3 0x6ed9eba1

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

__device__ void NTLM(char *, int, char*);
__device__ __constant__ char itoa16[17] = "0123456789ABCDEF";

__global__ void NTBruteforce(char *hex_format){
    char test[4] = {'t', 'h', 'e', 'n'};
    NTLM(test, 4, hex_format);      
}

__device__ void NTLM(char *key, int key_length, char *hex_format) {
    unsigned int nt_buffer[16] = { 0 };
    unsigned int output[4] = { 0 };

    // Globals for rounds
    unsigned int a = INIT_A;
    unsigned int b = INIT_B;
    unsigned int c = INIT_C;
    unsigned int d = INIT_D;

    // Prepare the string for hash calculation
    int i;
    int length = key_length;

    for (i = 0; i < length / 2; i++)
        nt_buffer[i] = key[2 * i] | (key[2 * i + 1] << 16);

    // Padding
    if (length % 2 == 1)
        nt_buffer[i] = key[length - 1] | 0x800000;
    else
        nt_buffer[i] = 0x80;

    // Put the length
    nt_buffer[14] = length << 4;

    // NTLM hash calculation

    /* Round 1 */
    a += (d ^ (b & (c ^ d))) + nt_buffer[0];
    a = (a << 3) | (a >> 29);
    d += (c ^ (a & (b ^ c))) + nt_buffer[1];
    d = (d << 7) | (d >> 25);
    c += (b ^ (d & (a ^ b))) + nt_buffer[2];
    c = (c << 11) | (c >> 21);
    b += (a ^ (c & (d ^ a))) + nt_buffer[3];
    b = (b << 19) | (b >> 13);

    a += (d ^ (b & (c ^ d))) + nt_buffer[4];
    a = (a << 3) | (a >> 29);
    d += (c ^ (a & (b ^ c))) + nt_buffer[5];
    d = (d << 7) | (d >> 25);
    c += (b ^ (d & (a ^ b))) + nt_buffer[6];
    c = (c << 11) | (c >> 21);
    b += (a ^ (c & (d ^ a))) + nt_buffer[7];
    b = (b << 19) | (b >> 13);

    a += (d ^ (b & (c ^ d))) + nt_buffer[8];
    a = (a << 3) | (a >> 29);
    d += (c ^ (a & (b ^ c))) + nt_buffer[9];
    d = (d << 7) | (d >> 25);
    c += (b ^ (d & (a ^ b))) + nt_buffer[10];
    c = (c << 11) | (c >> 21);
    b += (a ^ (c & (d ^ a))) + nt_buffer[11];
    b = (b << 19) | (b >> 13);

    a += (d ^ (b & (c ^ d))) + nt_buffer[12];
    a = (a << 3) | (a >> 29);
    d += (c ^ (a & (b ^ c))) + nt_buffer[13];
    d = (d << 7) | (d >> 25);
    c += (b ^ (d & (a ^ b))) + nt_buffer[14];
    c = (c << 11) | (c >> 21);
    b += (a ^ (c & (d ^ a))) + nt_buffer[15];
    b = (b << 19) | (b >> 13);

    /* Round 2 */
    a += ((b & (c | d)) | (c & d)) + nt_buffer[0] + SQRT_2;
    a = (a << 3) | (a >> 29);
    d += ((a & (b | c)) | (b & c)) + nt_buffer[4] + SQRT_2;
    d = (d << 5) | (d >> 27);
    c += ((d & (a | b)) | (a & b)) + nt_buffer[8] + SQRT_2;
    c = (c << 9) | (c >> 23);
    b += ((c & (d | a)) | (d & a)) + nt_buffer[12] + SQRT_2;
    b = (b << 13) | (b >> 19);

    a += ((b & (c | d)) | (c & d)) + nt_buffer[1] + SQRT_2;
    a = (a << 3) | (a >> 29);
    d += ((a & (b | c)) | (b & c)) + nt_buffer[5] + SQRT_2;
    d = (d << 5) | (d >> 27);
    c += ((d & (a | b)) | (a & b)) + nt_buffer[9] + SQRT_2;
    c = (c << 9) | (c >> 23);
    b += ((c & (d | a)) | (d & a)) + nt_buffer[13] + SQRT_2;
    b = (b << 13) | (b >> 19);

    a += ((b & (c | d)) | (c & d)) + nt_buffer[2] + SQRT_2;
    a = (a << 3) | (a >> 29);
    d += ((a & (b | c)) | (b & c)) + nt_buffer[6] + SQRT_2;
    d = (d << 5) | (d >> 27);
    c += ((d & (a | b)) | (a & b)) + nt_buffer[10] + SQRT_2;
    c = (c << 9) | (c >> 23);
    b += ((c & (d | a)) | (d & a)) + nt_buffer[14] + SQRT_2;
    b = (b << 13) | (b >> 19);

    a += ((b & (c | d)) | (c & d)) + nt_buffer[3] + SQRT_2;
    a = (a << 3) | (a >> 29);
    d += ((a & (b | c)) | (b & c)) + nt_buffer[7] + SQRT_2;
    d = (d << 5) | (d >> 27);
    c += ((d & (a | b)) | (a & b)) + nt_buffer[11] + SQRT_2;
    c = (c << 9) | (c >> 23);
    b += ((c & (d | a)) | (d & a)) + nt_buffer[15] + SQRT_2;
    b = (b << 13) | (b >> 19);

    /* Round 3 */
    a += (d ^ c ^ b) + nt_buffer[0] + SQRT_3;
    a = (a << 3) | (a >> 29);
    d += (c ^ b ^ a) + nt_buffer[8] + SQRT_3;
    d = (d << 9) | (d >> 23);
    c += (b ^ a ^ d) + nt_buffer[4] + SQRT_3;
    c = (c << 11) | (c >> 21);
    b += (a ^ d ^ c) + nt_buffer[12] + SQRT_3;
    b = (b << 15) | (b >> 17);

    a += (d ^ c ^ b) + nt_buffer[2] + SQRT_3;
    a = (a << 3) | (a >> 29);
    d += (c ^ b ^ a) + nt_buffer[10] + SQRT_3;
    d = (d << 9) | (d >> 23);
    c += (b ^ a ^ d) + nt_buffer[6] + SQRT_3;
    c = (c << 11) | (c >> 21);
    b += (a ^ d ^ c) + nt_buffer[14] + SQRT_3;
    b = (b << 15) | (b >> 17);

    a += (d ^ c ^ b) + nt_buffer[1] + SQRT_3;
    a = (a << 3) | (a >> 29);
    d += (c ^ b ^ a) + nt_buffer[9] + SQRT_3;
    d = (d << 9) | (d >> 23);
    c += (b ^ a ^ d) + nt_buffer[5] + SQRT_3;
    c = (c << 11) | (c >> 21);
    b += (a ^ d ^ c) + nt_buffer[13] + SQRT_3;
    b = (b << 15) | (b >> 17);

    a += (d ^ c ^ b) + nt_buffer[3] + SQRT_3;
    a = (a << 3) | (a >> 29);
    d += (c ^ b ^ a) + nt_buffer[11] + SQRT_3;
    d = (d << 9) | (d >> 23);
    c += (b ^ a ^ d) + nt_buffer[7] + SQRT_3;
    c = (c << 11) | (c >> 21);
    b += (a ^ d ^ c) + nt_buffer[15] + SQRT_3;
    b = (b << 15) | (b >> 17);

    output[0] = a + 0x67452301;
    output[1] = b + 0xefcdab89;
    output[2] = c + 0x98badcfe;
    output[3] = d + 0x10325476;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Convert the hash to hex (for being readable)
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for(i=0; i<4; i++)
    {
        int j = 0;
        unsigned int n = output[i];

        // Iterate the bytes of the integer
        for(; j<4; j++)
        {
            unsigned int convert = n % 256;
            hex_format[i * 8 + j * 2 + 1] = itoa16[convert % 16];
            convert = convert / 16;
            hex_format[i * 8 + j * 2 + 0] = itoa16[convert % 16];
            n = n / 256;
        }
    }       
}


int main()
{
    char* d_hex;
    char h_hex[33] = "";

    CUDA_SAFE_CALL(cudaMalloc(&d_hex, 33 * sizeof(char)));

    NTBruteforce<<<1, 1>>>(d_hex);

    CUDA_CHECK_ERROR();

    CUDA_SAFE_CALL(cudaMemcpy(h_hex, d_hex, 32 * sizeof(char), cudaMemcpyDeviceToHost)); 
    CUDA_SAFE_CALL(cudaFree(d_hex));
    
    h_hex[32] = '\0';
    std::cout << h_hex << std::endl;
}
