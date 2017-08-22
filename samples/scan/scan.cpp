/***************************************************************************
 *
 *  Copyright (C) 2017 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  scan.cpp
 *
 *  Description:
 *    Example of a parallel inclusive scan in SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

// The identity element for a given operation.
template <typename T, typename Op>
struct identity {};

template <typename T>
struct identity<T, std::plus<T>> {
  static constexpr T value = 0;
};

template <typename T>
struct identity<T, std::multiplies<T>> {
  static constexpr T value = 1;
};

template <typename T>
struct identity<T, std::logical_or<T>> {
  static constexpr T value = false;
};

template <typename T>
struct identity<T, std::logical_and<T>> {
  static constexpr T value = true;
};

/* Performs an inclusive scan with the given associative binary operation `Op`
 * in parallel using the provided accelerated hardware queue. Returns the result
 * as a vector of the same size as the input, and the input is unmodified.
 * Input size has to be a power of two. If the size isn't so, the input can
 * easily be padded to the nearest power of two with any values, and the scan on
 * the meaningful part of the data will stay the same. */
template <typename T, typename Op>
std::vector<T> par_scan(std::vector<T> const& in, sycl::queue& q) {
  size_t size = in.size();
  std::vector<T> out(size);

  {
    /* Input is sourced from `in`, but the output is written to `out`
     * using an overriden final data location. */
    sycl::buffer<T, 1> buf(in.data(), sycl::range<1>(size));
    buf.set_final_data(out.data());

    q.submit([&](sycl::handler& cgh) {
      auto data = buf.template get_access<sycl::access::mode::read_write>(cgh);
      sycl::accessor<T, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          /* This has one more element so that we can shift elements by 1
           * at the end of the kernel. */
          temp(size + 1, cgh);

      // Use identity struct as the unique kernel name.
      cgh.parallel_for<identity<T, Op>>(
          sycl::nd_range<1>(size / 2, size / 2), [=](sycl::nd_item<1> item) {
            /* Two-phase exclusive scan algorithm due to Guy E. Blelloch in
             * "Prefix Sums and Their Applications", 1990. */

            size_t gid = item.get_global_linear_id();

            // Read data into local memory.
            temp[2 * gid] = data[2 * gid];
            temp[2 * gid + 1] = data[2 * gid + 1];

            /* Perform partial reduction (up-sweep) on the data. The `off`
             * variable is 2 to the power of the current depth of the
             * reduction tree. In the paper, this corresponds to 2^d. */
            for (size_t off = 1; off < size; off *= 2) {
              // Synchronize local memory to observe the previous writes.
              item.barrier(sycl::access::fence_space::local_space);

              size_t i = gid * off * 2;
              if (i < size) {
                temp[i + off * 2 - 1] =
                    Op{}(temp[i + off * 2 - 1], temp[i + off - 1]);
              }
            }

            // Clear the last element to the identity before down-sweeping.
            if (gid == 0) {
              temp[size - 1] = identity<T, Op>::value;
            }

            /* Perform down-sweep on the tree to compute the whole scan.
             * Again, `off` is 2^d. */
            for (size_t off = size / 2; off > 0; off >>= 1) {
              item.barrier(sycl::access::fence_space::local_space);

              size_t i = gid * off * 2;
              if (i < size) {
                auto t = temp[i + off - 1];
                auto u = temp[i + off * 2 - 1];
                temp[i + off - 1] = u;
                temp[i + off * 2 - 1] = Op{}(t, u);
              }
            }

            // Synchronize again to observe results.
            item.barrier(sycl::access::fence_space::local_space);

            /* To return an inclusive rather than exclusive scan result, shift
             * each element left by 1 when writing back into global memory. */
            data[2 * gid] = temp[2 * gid + 1];
            data[2 * gid + 1] = temp[2 * gid + 2];
          });
    });
  }

  // Compute value of last scan which was not included in exclusive scan.
  out[size - 1] = Op{}(out[size - 2], in[size - 1]);

  return out;
}

constexpr size_t SIZE = 16;

/* Tests the scan with an addition operation, which is its most common use.
 * Returns 0 if successful, a nonzero value otherwise. */
int test_sum(sycl::queue& q) {
  // Initializes a vector of sequentially increasing values.
  std::vector<int32_t> in(SIZE);
  std::iota(in.begin(), in.end(), 1);

  // Compute the prefix sum using SYCL.
  auto sum = par_scan<int32_t, std::plus<int32_t>>(in, q);

  // Compute the same operation using the standard library.
  std::vector<int32_t> test_sum(SIZE);
  std::partial_sum(in.begin(), in.end(), test_sum.begin());

  // Check if the results are correct.
  auto equal = std::equal(sum.begin(), sum.end(), test_sum.begin());
  if (!equal) {
    std::cout << "SYCL sum computation incorrect! CPU Results:\n";
    for (auto a : test_sum) {
      std::cout << a << " ";
    }
    std::cout << "\nSYCL results:\n";
    for (auto a : sum) {
      std::cout << a << " ";
    }
    std::cout << std::endl;
    return 1;
  }

  return 0;
}

/* Tests the scan with a multiply operation, which is a sequence of factorials.
 * Returns 0 if successful, a nonzero value otherwise. */
int test_factorial(sycl::queue& q) {
  // Initializes a vector of sequentially increasing values.
  std::vector<int64_t> in(SIZE);
  std::iota(in.begin(), in.end(), 1);

  // Compute a sequence of factorials using SYCL.
  auto fact = par_scan<int64_t, std::multiplies<int64_t>>(in, q);

  // Compute the same operation using the standard library.
  std::vector<int64_t> test_fact(SIZE);
  std::partial_sum(in.begin(), in.end(), test_fact.begin(),
                   std::multiplies<int64_t>{});

  // Check if the results are correct.
  auto equal = std::equal(fact.begin(), fact.end(), test_fact.begin());
  if (!equal) {
    std::cout << "SYCL factorial computation incorrect! CPU Results:\n";
    for (auto a : test_fact) {
      std::cout << a << " ";
    }
    std::cout << "\nSYCL results:\n";
    for (auto a : fact) {
      std::cout << a << " ";
    }
    std::cout << std::endl;
    return 1;
  }

  return 0;
}

int main() {
  sycl::queue q{sycl::default_selector{}};

  auto ret = test_sum(q);
  if (ret != 0) {
    return ret;
  }
  ret = test_factorial(q);
  if (ret != 0) {
    return ret;
  }

  std::cout << "Results are correct." << std::endl;
  return 0;
}
