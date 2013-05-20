#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

const int N = 1024*1024;

struct is_positive
{
__host__ __device__
bool operator()(const int &x)
{
  return x >= 0;
}
};

struct my_compare
{
  __device__ __host__ bool operator()(const float x, const float y) const
  {
    return !((x<0.0f) && (y>0.0f));
  }
};

void print_vec(const thrust::host_vector<int>& v)
{
  for(size_t i = 0; i < min((int)v.size(), 10); i++)
    std::cout << "  " << v[i];
  std::cout << "\n";
}

void partition_test ()
{
  std::cout << "### PARTITION VERSION ###" << std::endl;

  thrust::host_vector<int> keyVec(N);
  thrust::host_vector<int> valVec(N);

  int sign = 1;
  for(int i = 0; i < N; ++i)
  {
    keyVec[i] = sign * i;
    valVec[i] = i;
    sign *= -1;
  }

  // Copy host to device
  thrust::device_vector<int> d_keyVec = keyVec;
  thrust::device_vector<int> d_valVec = valVec;

  std::cout << "Before:\n  keyVec = ";
  print_vec(keyVec);
  std::cout << "  valVec = ";
  print_vec(valVec);

  // Partition key-val on device
  thrust::partition(thrust::make_zip_iterator(thrust::make_tuple(d_keyVec.begin(), d_valVec.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(d_keyVec.end(), d_valVec.end())),
                    d_keyVec.begin(),
                    is_positive());
                    
  // Copy result back to host
  keyVec = d_keyVec;
  valVec = d_valVec;

  std::cout << "After:\n  keyVec = ";
  print_vec(keyVec);
  std::cout << "  valVec = ";
  print_vec(valVec);
  std::cout << std::endl;
}

void sort_test ()
{
  std::cout << "### SORT VERSION ###" << std::endl;

  thrust::host_vector<int> keyVec(N);
  thrust::host_vector<int> valVec(N);

  int sign = 1;
  for(int i = 0; i < N; ++i)
  {
    keyVec[i] = sign * i;
    valVec[i] = i;
    sign *= -1;
  }
  
  // Copy host to device
  thrust::device_vector<int> d_keyVec = keyVec;
  thrust::device_vector<int> d_valVec = valVec;

  std::cout << "Before:\n  keyVec = ";
  print_vec(keyVec);
  std::cout << "  valVec = ";
  print_vec(valVec);

  // Sort key-val on device
  thrust::sort_by_key(d_keyVec.begin(), d_keyVec.end(),
                      d_valVec.begin(), my_compare());
                    
  // Copy result back to host
  keyVec = d_keyVec;
  valVec = d_valVec;

  std::cout << "After:\n  keyVec = ";
  print_vec(keyVec);
  std::cout << "  valVec = ";
  print_vec(valVec);
  std::cout << std::endl;
}

int main()
{
  sort_test();
  partition_test();
}
