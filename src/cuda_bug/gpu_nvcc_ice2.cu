/* vim:set ts=4 sw=4 noexpandtab: */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

const int N = 35;

// Wrong dimensions here can lead to Thrust compilation error
const uint SIZE_1 = 3;
const uint SIZE_2 = 5;
const uint SIZE_3 = 32;
const uint SIZE_4 = 3;

typedef float type_t;

template <typename T, uint N1, uint N2>
struct Array2D
{
  T val[N1][N2];
};

template <typename T, uint N1, uint N2, uint N3>
struct Array3D
{
  T val[N1][N2][N3];
};

template <typename T, uint N1, uint N2, uint N3, uint N4>
struct Array4D
{
  T val[N1][N2][N3][N4];
};

template <typename T, uint N1, uint N2>
struct array2D_plus : public thrust::binary_function
    <Array2D<T,N1,N2>, Array2D<T,N1,N2>, Array2D<T,N1,N2> >
{
  __host__ __device__
  Array2D<T,N1,N2> operator()(const Array2D<T,N1,N2>& x, const Array2D<T,N1,N2>& y)
  {
    Array2D<T,N1,N2> res;

    for (size_t i = 0; i < N1; i++)
      for (size_t j = 0; j < N2; j++)
        res.val[i][j] = x.val[i][j] + y.val[i][j];

    return res;
  }
};

template <typename T, uint N1, uint N2>
void print_array(const Array2D<T,N1,N2>& a)
{
  for (size_t i = 0; i < N1; i++)
    for (size_t j = 0; j < N2; j++)
      std::cout << "  " << a.val[i][j];
  std::cout << "\n";
}

template <typename T, uint N1, uint N2>
void print_array(const thrust::host_vector<Array2D<T,N1,N2> >& v)
{
  for (size_t i = 0; i < v.size (); ++i)
  {
    print_array (v[i]);
    std::cout << std::endl;
  }
}

template <typename T, uint N1, uint N2, uint N3>
struct array3D_plus : public thrust::binary_function
    <Array3D<T,N1,N2,N3>, Array3D<T,N1,N2,N3>, Array3D<T,N1,N2,N3> >
{
  __host__ __device__
  Array3D<T,N1,N2,N3> operator()(const Array3D<T,N1,N2,N3>& x,
                                 const Array3D<T,N1,N2,N3>& y)
  {
    Array3D<T,N1,N2,N3> res;

    for (size_t i = 0; i < N1; i++)
      for (size_t j = 0; j < N2; j++)
        for (size_t k = 0; k < N3; k++)
          res.val[i][j][k] = x.val[i][j][k] + y.val[i][j][k];

    return res;
  }
};

template <typename T, uint N1, uint N2, uint N3>
void print_array(const Array3D<T,N1,N2,N3>& a)
{
  for (size_t i = 0; i < N1; i++)
    for (size_t j = 0; j < N2; j++)
      for (size_t k = 0; k < N3; k++)
        std::cout << "  " << a.val[i][j][k];
  std::cout << "\n";
}

template <typename T, uint N1, uint N2, uint N3>
void print_array(const thrust::host_vector<Array3D<T,N1,N2,N3> >& v)
{
  for (size_t i = 0; i < v.size (); ++i)
  {
    print_array (v[i]);
    std::cout << std::endl;
  }
}


template <typename T, uint N1, uint N2, uint N3, uint N4>
struct array4D_plus : public thrust::binary_function
    <Array4D<T,N1,N2,N3,N4>, Array4D<T,N1,N2,N3,N4>, Array4D<T,N1,N2,N3,N4> >
{
  __host__ __device__
  Array4D<T,N1,N2,N3,N4> operator()(const Array4D<T,N1,N2,N3,N4>& x,
                                    const Array4D<T,N1,N2,N3,N4>& y)
  {
    Array4D<T,N1,N2,N3,N4> res;

    for (size_t i = 0; i < N1; i++)
      for (size_t j = 0; j < N2; j++)
        for (size_t k = 0; k < N3; k++)
          for (size_t r = 0; r < N4; r++)
            res.val[i][j][k][r] = x.val[i][j][k][r] + y.val[i][j][k][r];

    return res;
  }
};


template <typename T, uint N1, uint N2, uint N3, uint N4>
void print_array(const Array4D<T,N1,N2,N3,N4>& a)
{
  for (size_t i = 0; i < N1; i++)
    for (size_t j = 0; j < N2; j++)
      for (size_t k = 0; k < N3; k++)
        for (size_t r = 0; r < N4; r++)
          std::cout << "  " << a.val[i][j][k][r];
  std::cout << "\n";
}

template <typename T, uint N1, uint N2, uint N3, uint N4>
void print_array(const thrust::host_vector<Array4D<T,N1,N2,N3,N4> >& v)
{
  for (size_t i = 0; i < v.size (); ++i)
  {
    print_array (v[i]);
    std::cout << std::endl;
  }
}

void scan_test_2D ()
{
  std::cout << "### SCAN 2D VERSION ###" << std::endl;

  thrust::host_vector<Array2D<type_t,SIZE_1,SIZE_2> > h_vec(N);

  for (uint i = 0; i < N; ++i)
    for (uint j = 0; j < SIZE_1; ++j)
      for (uint k = 0; k < SIZE_2; ++k)
      {
        h_vec[i].val[j][k] = k;
      }

  // Copy host to device
  thrust::device_vector<Array2D<type_t,SIZE_1,SIZE_2> > d_vec = h_vec;

  std::cout << "Before:\n";
  print_array(h_vec);

  // In-place scan
  array2D_plus<type_t,SIZE_1,SIZE_2> binary_op;
  thrust::device_vector<Array2D<type_t,SIZE_1,SIZE_2> >::iterator begin = d_vec.begin();
  thrust::device_vector<Array2D<type_t,SIZE_1,SIZE_2> >::iterator end = d_vec.end();
  thrust::inclusive_scan(begin, end, begin, binary_op);

  // Copy result back to host
  h_vec = d_vec;

  std::cout << "After:\n";
  print_array(h_vec);
  std::cout << std::endl;
}


void scan_test_3D ()
{
  std::cout << "### SCAN 3D VERSION ###" << std::endl;

  typedef Array3D<type_t,SIZE_3,SIZE_2,SIZE_1> array_t;
  thrust::host_vector<array_t> h_vec(N);

  for (uint i = 0; i < N; ++i)
    for (uint j = 0; j < SIZE_3; ++j)
      for (uint k = 0; k < SIZE_2; ++k)
        for (uint r = 0; r < SIZE_1; ++r)
        {
          h_vec[i].val[j][k][r] = r;
        }

  // Copy host to device
  thrust::device_vector<array_t> d_vec = h_vec;

  std::cout << "Before:\n";
  print_array(h_vec);

  // In-place scan
  array3D_plus<type_t,SIZE_3,SIZE_2,SIZE_1> binary_op;
  thrust::device_vector<array_t>::iterator begin = d_vec.begin();
  thrust::device_vector<array_t>::iterator end = d_vec.end();
  thrust::inclusive_scan(begin, end, begin, binary_op);

  // Copy result back to host
  h_vec = d_vec;

  std::cout << "After:\n";
  print_array(h_vec);
  std::cout << std::endl;
}


void scan_test_4D ()
{
  std::cout << "### SCAN 4D VERSION ###" << std::endl;

  typedef Array4D<type_t,SIZE_4,SIZE_3,SIZE_2,SIZE_1> array_t;
  thrust::host_vector<array_t> h_vec(N);

  for (uint i = 0; i < N; ++i)
    for (uint j = 0; j < SIZE_4; ++j)
      for (uint k = 0; k < SIZE_3; ++k)
        for (uint r = 0; r < SIZE_2; ++r)
          for (uint s = 0; s < SIZE_1; ++s)
          {
            h_vec[i].val[j][k][r][s] = s;
          }

  // Copy host to device
  thrust::device_vector<array_t> d_vec = h_vec;

  std::cout << "Before:\n";
  print_array(h_vec);

  // In-place scan
  array4D_plus<type_t,SIZE_4,SIZE_3,SIZE_2,SIZE_1> binary_op;
  thrust::device_vector<array_t>::iterator begin = d_vec.begin();
  thrust::device_vector<array_t>::iterator end = d_vec.end();
  thrust::inclusive_scan(begin, end, begin, binary_op);

  // Copy result back to host
  h_vec = d_vec;

  std::cout << "After:\n";
  print_array(h_vec);
  std::cout << std::endl;
}


int main()
{
  scan_test_2D();
  scan_test_3D();
  scan_test_4D();
}
