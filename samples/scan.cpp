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

// Dummy struct to generate unique kernel name types
template <typename T, typename U, typename V>
struct kernel_name {};

/* Performs an inclusive scan with the given associative binary operation `Op`
 * on the data in the `in` buffer. Runs in parallel on the provided accelerated
 * hardware queue. Modifies the input buffer to contain the results of the scan.
 * Input size has to be a power of two. If the size isn't so, the input can
 * easily be padded to the nearest power of two with any values, and the scan on
 * the meaningful part of the data will stay the same. */
template <typename T, typename Op>
void par_scan(sycl::buffer<T, 1>& in, sycl::queue& q) {
  if ((in.get_count() & (in.get_count() - 1)) != 0 || in.get_count() == 0) {
    throw std::runtime_error("Given input size is not a power of two.");
  }

  // Retrieve the device associated with the given queue.
  auto dev = q.get_device();

  // Check if there is enough global memory.
  size_t global_mem_size = dev.get_info<sycl::info::device::global_mem_size>();
  if (!dev.is_host() && in.get_count() > (global_mem_size / 2)) {
    throw std::runtime_error("Input size exceeds device global memory size.");
  }

  /* Check if local memory is available. On host no local memory is fine, since
   * it is emulated. */
  if (!dev.is_host() && dev.get_info<sycl::info::device::local_mem_type>() ==
                            sycl::info::local_mem_type::none) {
    throw std::runtime_error("Device does not have local memory.");
  }

  // Obtain device limits.
  size_t max_wgroup_size =
      dev.get_info<sycl::info::device::max_work_group_size>();
  size_t local_mem_size = dev.get_info<sycl::info::device::local_mem_size>();

  /* Find a work-group size that is guaranteed to fit in local memory and is
   * below the maximum work-group size of the device. */
  size_t wgroup_size_lim =
      sycl::min(max_wgroup_size, local_mem_size / (2 * sizeof(T)));

  /* Every work-item processes two elements, so the work-group size has to
   * divide this number evenly. */
  size_t half_in_size = in.get_count() / 2;

  size_t wgroup_size = 0;
  /* Find the largest power of two that divides half_in_size and is within the
   * device limit. */
  for (size_t pow = size_t(1) << (sizeof(size_t) * 8 - 1); pow > 0; pow >>= 1) {
    if ((half_in_size / pow) * pow == half_in_size && pow <= wgroup_size_lim) {
      wgroup_size = pow;
      break;
    }
  }

  if (wgroup_size == 0) {
    throw std::runtime_error(
        "Could not find an appropriate work-group size for the given input.");
  }

  q.submit([&](sycl::handler& cgh) {
    auto data = in.template get_access<sycl::access::mode::read_write>(cgh);
    sycl::accessor<T, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>

        temp(wgroup_size * 2, cgh);

    // Use dummy struct as the unique kernel name.
    cgh.parallel_for<kernel_name<T, Op, class scan_segments>>(
        sycl::nd_range<1>(half_in_size, wgroup_size),
        [=](sycl::nd_item<1> item) {
          /* Two-phase exclusive scan algorithm due to Guy E. Blelloch in
           * "Prefix Sums and Their Applications", 1990. */

          size_t gid = item.get_global_linear_id();
          size_t lid = item.get_local_linear_id();

          // Read data into local memory.
          temp[2 * lid] = data[2 * gid];
          temp[2 * lid + 1] = data[2 * gid + 1];

          // Preserve the second input element to add at the end.
          auto second_in = temp[2 * lid + 1];

          /* Perform partial reduction (up-sweep) on the data. The `off`
           * variable is 2 to the power of the current depth of the
           * reduction tree. In the paper, this corresponds to 2^d. */
          for (size_t off = 1; off < (wgroup_size * 2); off *= 2) {
            // Synchronize local memory to observe the previous writes.
            item.barrier(sycl::access::fence_space::local_space);

            size_t i = lid * off * 2;
            if (i < wgroup_size * 2) {
              temp[i + off * 2 - 1] =
                  Op{}(temp[i + off * 2 - 1], temp[i + off - 1]);
            }
          }

          // Clear the last element to the identity before down-sweeping.
          if (lid == 0) {
            temp[wgroup_size * 2 - 1] = identity<T, Op>::value;
          }

          /* Perform down-sweep on the tree to compute the whole scan.
           * Again, `off` is 2^d. */
          for (size_t off = wgroup_size; off > 0; off >>= 1) {
            item.barrier(sycl::access::fence_space::local_space);

            size_t i = lid * off * 2;
            if (i < wgroup_size * 2) {
              auto t = temp[i + off - 1];
              auto u = temp[i + off * 2 - 1];
              temp[i + off - 1] = u;
              temp[i + off * 2 - 1] = Op{}(t, u);
            }
          }

          // Synchronize again to observe results.
          item.barrier(sycl::access::fence_space::local_space);

          /* To return an inclusive rather than exclusive scan result, shift
           * each element left by 1 when writing back into global memory. If
           * we are the last work-item, also add on the final element. */
          data[2 * gid] = temp[2 * lid + 1];

          if (lid == wgroup_size - 1) {
            data[2 * gid + 1] = Op{}(temp[2 * lid + 1], second_in);
          } else {
            data[2 * gid + 1] = temp[2 * lid + 2];
          }
        });
  });

  // At this point we have computed the inclusive scans of this many segments.
  size_t n_segments = half_in_size / wgroup_size;

  if (n_segments == 1) {
    // If all of the data is in one segment, we're done.
    return;
  }
  // Otherwise we have to propagate the scan results forward into later
  // segments.

  // Allocate space for one (last) element per segment.
  sycl::buffer<T, 1> ends{sycl::range<1>(n_segments)};

  // Store the elements in this space.
  q.submit([&](sycl::handler& cgh) {
    auto scans = in.template get_access<sycl::access::mode::read>(cgh);
    auto elems =
        ends.template get_access<sycl::access::mode::discard_write>(cgh);

    cgh.parallel_for<kernel_name<T, Op, class copy_ends>>(
        sycl::range<1>(n_segments), [=](sycl::item<1> item) {
          auto id = item.get_linear_id();
          // Offset into the last element of each segment.
          elems[item] = scans[(id + 1) * 2 * wgroup_size - 1];
        });
  });

  // Recursively scan the array of last elements.
  par_scan<T, Op>(ends, q);

  // Add the results of the scan to each segment.
  q.submit([&](sycl::handler& cgh) {
    auto ends_scan = ends.template get_access<sycl::access::mode::read>(cgh);
    auto data = in.template get_access<sycl::access::mode::read_write>(cgh);

    cgh.parallel_for<kernel_name<T, Op, class add_ends>>(
        // Work with one less work-group, since the first segment is correct.
        sycl::nd_range<1>(half_in_size - wgroup_size, wgroup_size),
        [=](sycl::nd_item<1> item) {
          auto group = item.get_group_linear_id();

          // Start with the second segment.
          auto off_gid = item.get_global_linear_id() + wgroup_size;

          /* Each work-group adds the corresponding number in the
           * "last element scan" array to every element in the group's
           * segment. */
          data[off_gid * 2] = Op{}(data[off_gid * 2], ends_scan[group]);
          data[off_gid * 2 + 1] = Op{}(data[off_gid * 2 + 1], ends_scan[group]);
        });
  });
}

/* Tests the scan with an addition operation, which is its most common use.
 * Returns 0 if successful, a nonzero value otherwise. */
int test_sum(sycl::queue& q) {
  constexpr size_t size = 512;

  // Initializes a vector of sequentially increasing values.
  std::vector<int32_t> in(size);
  std::iota(in.begin(), in.end(), 1);

  // Compute the prefix sum using SYCL.
  std::vector<int32_t> sum(in.size());
  {
    // Read from `in`, but write into `sum`.
    sycl::buffer<int32_t, 1> buf(sycl::range<1>(in.size()));
    buf.set_final_data(sum.data());
    q.submit([&](sycl::handler& cgh) {
      auto acc = buf.get_access<sycl::access::mode::write>(cgh);
      cgh.copy(in.data(), acc);
    });

    par_scan<int32_t, std::plus<int32_t>>(buf, q);
  }

  // Compute the same operation using the standard library.
  std::vector<int32_t> test_sum(in.size());
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
  // Anything above this size overflows the int64_t type
  constexpr size_t size = 16;

  // Initializes a vector of sequentially increasing values.
  std::vector<int64_t> in(size);
  std::iota(in.begin(), in.end(), 1);

  // Compute a sequence of factorials using SYCL.
  std::vector<int64_t> fact(in.size());
  {
    // Read from `in`, but write into `fact`.
    sycl::buffer<int64_t, 1> buf(sycl::range<1>(in.size()));
    buf.set_final_data(fact.data());
    q.submit([&](sycl::handler& cgh) {
      auto acc = buf.get_access<sycl::access::mode::write>(cgh);
      cgh.copy(in.data(), acc);
    });

    par_scan<int64_t, std::multiplies<int64_t>>(buf, q);
  }

  // Compute the same operation using the standard library.
  std::vector<int64_t> test_fact(in.size());
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
