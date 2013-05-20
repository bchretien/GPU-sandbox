#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform_reduce.h>

const int N = 10;

struct diff_char
{
__host__ __device__
int operator()(const thrust::tuple<char,char> &t)
{
  return thrust::get<0>(t) != thrust::get<1>(t);
}
};

template <typename T>
void print_vec(const thrust::host_vector<T>& v)
{
  for(size_t i = 0; i < min((T)v.size(), 10); i++)
    std::cout << "  " << v[i];
  std::cout << "\n";
}

void hamming_test ()
{
  typedef char type_t;
  thrust::host_vector<type_t> h_A(N);
  thrust::host_vector<type_t> h_B(N);
  thrust::host_vector<int> h_diff(N);
  
  char* word_1 = "azertyuiop";
  char* word_2 = "qwertyuior";

  srand(time(NULL));

  for(int i = 0; i < N; ++i)
  {
    h_A[i] = word_1[i];
    h_B[i] = word_2[i];
    h_diff[i] = true;
  }

  // Copy host to device
  thrust::device_vector<type_t> d_A = h_A;
  thrust::device_vector<type_t> d_B = h_B;
  thrust::device_vector<int> d_diff = h_diff;

  std::cout << "A = ";
  print_vec(h_A);
  std::cout << "B = ";
  print_vec(h_B);

  // difference between A and B
  thrust::transform(d_A.begin(), d_A.end(),
                    d_B.begin(),
                    d_diff.begin(),
                    thrust::not_equal_to<type_t>());
                    
  int hamming_distance = thrust::transform_reduce(
    thrust::make_zip_iterator(thrust::make_tuple(d_A.begin(), d_B.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(d_A.end(), d_B.end())),
    diff_char(),
    0,
    thrust::plus<int>());
  
  h_diff = d_diff;

  std::cout << "Hamming_distance(A,B) = " << hamming_distance << std::endl;
  std::cout << "Differences = ";
  print_vec(h_diff);
  std::cout << std::endl;
}

int main()
{
  hamming_test();
}
