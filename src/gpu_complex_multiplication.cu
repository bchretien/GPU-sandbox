#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <time.h>

struct ElementWiseProductBasic : public thrust::binary_function<float2,float2,float2>
{
    __host__ __device__
    float2 operator()(const float2& v1, const float2& v2) const
    {
        float2 res;
        res.x = v1.x * v2.x - v1.y * v2.y;
        res.y = v1.x * v2.y + v1.y * v2.x;
        return res;
    }
};

/**
 * See: http://www.embedded.com/design/embedded/4007256/Digital-Signal-Processing-Tricks--Fast-multiplication-of-complex-numbers%5D
 */
struct ElementWiseProductModified : public thrust::binary_function<float2,float2,float2>
{
    __host__ __device__
    float2 operator()(const float2& v1, const float2& v2) const
    {
        float2 res;
        float a, b, c, d, k;
        a = v1.x;
        b = v1.y;
        c = v2.x;
        d = v2.y;
        k = a * (c + d);
        d =  d * (a + b);
        c =  c * (b - a);
        res.x = k -d;
        res.y = k + c;
        return res;
    }
};

int get_random_int(int min, int max)
{
    return min + (rand() % (int)(max - min + 1));
}

thrust::host_vector<float2> init_vector(const size_t N)
{
    thrust::host_vector<float2> temp(N);
    for(size_t i = 0; i < N; i++)
    {
        temp[i].x = get_random_int(0, 10);
        temp[i].y = get_random_int(0, 10);
    }
    return temp;
}

int main(void)
{
    const size_t N = 100000;
    const bool compute_basic_product    = true;
    const bool compute_modified_product = true;

    srand(time(NULL));

    thrust::host_vector<float2>   h_A = init_vector(N);
    thrust::host_vector<float2>   h_B = init_vector(N);
    thrust::device_vector<float2> d_A = h_A;
    thrust::device_vector<float2> d_B = h_B;

    thrust::host_vector<float2> h_result(N);
    thrust::host_vector<float2> h_result_modified(N);

    if (compute_basic_product)
    {
        thrust::device_vector<float2> d_result(N);

        thrust::transform(d_A.begin(), d_A.end(),
                          d_B.begin(), d_result.begin(),
                          ElementWiseProductBasic());
        h_result = d_result;
    }

    if (compute_modified_product)
    {
        thrust::device_vector<float2> d_result_modified(N);

        thrust::transform(d_A.begin(), d_A.end(),
                          d_B.begin(), d_result_modified.begin(),
                          ElementWiseProductModified());
        h_result_modified = d_result_modified;
    }

    std::cout << std::fixed;
    for (size_t i = 0; i < 4; i++)
    {
        float2 a = h_A[i];
        float2 b = h_B[i];

        std::cout << "(" << a.x << "," << a.y << ")";
        std::cout << " * ";
        std::cout << "(" << b.x << "," << b.y << ")";

        if (compute_basic_product)
        {
            float2 prod = h_result[i];
            std::cout << " = ";
            std::cout << "(" << prod.x << "," << prod.y << ")";
        }

        if (compute_modified_product)
        {
            float2 prod_modified = h_result_modified[i];
            std::cout << " = ";
            std::cout << "(" << prod_modified.x << "," << prod_modified.y << ")";
        }
        std::cout << std::endl;
    }   

    return 0;
}
