/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  scan-subgroups.cpp
 *
 *  Description:
 *    Example of a parallel inclusive scan in SYCL using subgroup operations.
 *
 **************************************************************************/

#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

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
void par_scan(sycl::buffer<T, 1> in, sycl::queue q) {
  if ((in.get_count() & (in.get_count() - 1)) != 0 || in.get_count() == 0) {
    throw std::runtime_error("Given input size is not a power of two.");
  }

  auto dev = q.get_device();

  // Check if there is enough global memory.
  size_t global_mem_size = dev.get_info<sycl::info::device::global_mem_size>();
  if (in.get_count() > global_mem_size) {
    throw std::runtime_error("Input size exceeds device global memory size.");
  }

  // Obtain device limits.
  size_t max_wgroup_size =
      dev.get_info<sycl::info::device::max_work_group_size>();
  size_t local_mem_size = dev.get_info<sycl::info::device::local_mem_size>();

  /* Find a work-group size that is guaranteed to fit in local memory and is
   * below the maximum work-group size of the device. */
  size_t wgroup_size_lim =
      sycl::min(max_wgroup_size, local_mem_size / sizeof(T));

  size_t input_size = in.get_count();

  size_t wgroup_size = 0;
  for (size_t pow = size_t(1) << (sizeof(size_t) * 8 - 1); pow > 0; pow >>= 1) {
    if ((input_size / pow) * pow == input_size && pow <= wgroup_size_lim) {
      wgroup_size = pow;
      break;
    }
  }

  if (wgroup_size == 0) {
    throw std::runtime_error(
        "Could not find an appropriate work-group size for the given input.");
  }

  using namespace sycl::access;
  q.submit([&](sycl::handler& cgh) {
    auto data = in.template get_access<mode::read_write>(cgh);
    auto temp = sycl::accessor<T, 1, mode::read_write, target::local>(
        sycl::range<1>(wgroup_size), cgh);
    cgh.parallel_for<kernel_name<T, Op, class scan_segments>>(
        sycl::nd_range<1>(input_size, wgroup_size), [=](sycl::nd_item<1> item) {
          size_t gid = item.get_global_linear_id();

          auto sub_group = item.get_sub_group();
          auto scan_res = inclusive_scan_over_group(sub_group, data[gid], Op{});
          if (sub_group.get_local_id() == sub_group.get_local_range() - 1) {
            temp[sub_group.get_group_linear_id()] = scan_res;
          }
          item.barrier(sycl::access::fence_space::local_space);
          for (auto i = 1u; i < sub_group.get_group_linear_range(); i++) {
            scan_res += sub_group.get_group_linear_id() >= i ? temp[i - 1] : 0;
          }
          data[gid] = scan_res;
        });
  });

  // At this point we have computed the inclusive scans of this many segments.
  size_t n_segments = input_size / wgroup_size;
  if (n_segments == 1) {
    // If all of the data is in one segment, we're done.
    return;
  }

  // Store the last element of each segment in a temporary buffer
  sycl::buffer<T, 1> ends{sycl::range<1>(n_segments)};
  q.submit([&](sycl::handler& cgh) {
    auto scans = in.template get_access<mode::read>(cgh);
    auto elems = ends.template get_access<mode::discard_write>(cgh);

    cgh.parallel_for<kernel_name<T, Op, class copy_ends>>(
        sycl::range<1>(n_segments), [=](sycl::item<1> item) {
          auto id = item.get_linear_id();
          // Offset into the last element of each segment.
          elems[item] = scans[(id + 1) * wgroup_size - 1];
        });
  });

  // Recursively scan the array of last elements.
  par_scan<T, Op>(ends, q);

  // Add the results of the scan to each segment.
  q.submit([&](sycl::handler& cgh) {
    auto ends_scan = ends.template get_access<mode::read>(cgh);
    auto data = in.template get_access<mode::read_write>(cgh);

    cgh.parallel_for<kernel_name<T, Op, class add_ends>>(
        // Work with one less work-group, since the first segment is correct.
        sycl::nd_range<1>(input_size - wgroup_size, wgroup_size),
        [=](sycl::nd_item<1> item) {
          auto group = item.get_group_linear_id();
          auto off_gid = item.get_global_linear_id() + wgroup_size;
          data[off_gid] = Op{}(data[off_gid], ends_scan[group]);
        });
  });
}

/* Tests the scan with an addition operation, which is its most common use.
 * Returns 0 if successful, a nonzero value otherwise. */
int test_sum(sycl::queue& q) {
  constexpr size_t size = 8192;

  std::vector<int32_t> in(size);
  std::iota(in.begin(), in.end(), 1);

  std::vector<int32_t> sum(in.size());
  {
    sycl::buffer<int32_t, 1> buf(sycl::range<1>(in.size()));
    buf.set_final_data(sum.data());
    q.submit([&](sycl::handler& cgh) {
      auto acc = buf.get_access<sycl::access::mode::write>(cgh);
      cgh.copy(in.data(), acc);
    });

    par_scan<int32_t, sycl::plus<int32_t>>(buf, q);
  }

  std::vector<int32_t> test_sum(in.size());
  std::partial_sum(in.begin(), in.end(), test_sum.begin());

  auto equal = std::equal(sum.begin(), sum.end(), test_sum.begin());
  if (!equal) {
    std::cout << "SYCL sum computation incorrect! CPU Results:\n";
    for (auto a : test_sum) {
      std::cout << a << "\n";
    }
    std::cout << "\nSYCL results:\n";
    for (auto a : sum) {
      std::cout << a << "\n";
    }
    std::cout << std::endl;
    return 1;
  }

  return 0;
}

int main() {
  sycl::queue q{sycl::default_selector{}};

  if (SYCL_LANGUAGE_VERSION < 202000) {
    std::cout << "This sample must be compiled with SYCL 2020 support\n";
    return 0;
  }

  auto ret = test_sum(q);
  if (ret != 0) {
    std::cout << "Results are not correct.\n";
    return ret;
  }

  std::cout << "Results are correct.\n";
  return 0;
}
