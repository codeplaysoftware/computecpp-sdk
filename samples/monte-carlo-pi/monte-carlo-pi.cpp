/***************************************************************************
 *
 *  Copyright (C) 2019 Codeplay Software Limited
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
 *  monte-carlo-pi.cpp
 *
 *  Description:
 *    Example of Monte-Carlo Pi approxamtion algorithm in SYCL. Also,
 *    demonstrating how to query the maximum number of work-items in a
 *    work-group to check if a kernel can be executed with the initially 
 *    desired work-group size.
 *
 *
 **************************************************************************/

#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <random>
#include <typeinfo>
#include <vector>

namespace sycl = cl::sycl;

// Monte-Carlo Pi SYCL C++ functor
class monte_carlo_pi_kernel {
  template <typename dataT>
  using read_const_accessor =
      sycl::accessor<dataT, 1, sycl::access::mode::read,
                     sycl::access::target::constant_buffer>;
  template <typename dataT>
  using write_global_accessor =
      sycl::accessor<dataT, 1, sycl::access::mode::write,
                     sycl::access::target::global_buffer>;
  template <typename dataT>
  using write_local_accessor =
      sycl::accessor<dataT, 1, sycl::access::mode::write,
                     sycl::access::target::local>;

 public:
  monte_carlo_pi_kernel(read_const_accessor<sycl::cl_float2> points_ptr,
                        write_global_accessor<sycl::cl_int> results_ptr,
                        write_local_accessor<sycl::cl_int> local_results_ptr)
      : _points_ptr(points_ptr),
        _results_ptr(results_ptr),
        _local_results_ptr(local_results_ptr) {}

  void operator()(sycl::nd_item<1> item) {
    // Get our id
    auto global_id = item.get_global_id(0);
    auto local_id = item.get_local_id(0);
    auto local_dim = item.get_local_range(0);
    auto group_id = item.get_group(0);

    // Get the point to work on
    sycl::float2 point = _points_ptr[global_id];

    // Calculate the length - built-in SYCL function
    // length: sqrt(point.x * point.x + point.y * point.y)
    auto len = sycl::length(point);

    // Result is either 1 or 0
    _local_results_ptr[local_id] = (len <= 1.0) ? 1 : 0;

    // Wait for the entire work group to get here.
    item.barrier(sycl::access::fence_space::local_space);

    // If work item 0 in work group sum local values
    if (local_id == 0) {
      int sum = 0;
      for (auto i = 0u; i < local_dim; ++i) {
        if (_local_results_ptr[i] == 1)
          ++sum;
      }
      // Store the sum in global memory
      _results_ptr[group_id] = sum;
    }
  }

 private:
  read_const_accessor<sycl::cl_float2> _points_ptr;
  write_global_accessor<sycl::cl_int> _results_ptr;
  write_local_accessor<sycl::cl_int> _local_results_ptr;
};

/* A helper to define a "perfect" work-group size dependant on selected device
 * and kernel maximum allowance */
size_t get_best_work_group_size(size_t work_group_size,
                                const sycl::device& device,
                                const sycl::kernel& kernel) {
  if (device.is_host()) {
    /* Maximum Work-Group Size on selected device.
     * (See: sycl-1.2.1.pdf: p.62, table 4.20) */
    const size_t max_device_work_group_size =
        device.get_info<cl::sycl::info::device::max_work_group_size>();
    /* Check if the desired work-group size will be allowed on the host device
     * and query the maximum possible size on that device in case the desired
     * one is more than the allowed */
    if (work_group_size > max_device_work_group_size) {
      std::cout << "Maximum work-group size for device "
                << device.get_info<cl::sycl::info::device::name>() << ": "
                << max_device_work_group_size << "\n\n";
      return max_device_work_group_size;
    }
    return work_group_size;
  } else {
    /* Maximum Work-Group Size for given kernel on selected device.
     * (See: sycl-1.2.1.pdf: p.180, table 4.85) */
    const size_t max_kernel_work_group_size = kernel.get_work_group_info<
        sycl::info::kernel_work_group::work_group_size>(device);
    /* Verify if the kernel can be executed with our desired work-group size,
     * and if it can't use the maximum allowed kernel work-group size for the
     * selected device.
     */
    if (work_group_size > max_kernel_work_group_size) {
      std::cout << "Maximum work-group size for "
                << typeid(monte_carlo_pi_kernel).name() << " on device "
                << device.get_info<sycl::info::device::name>() << ": "
                << max_kernel_work_group_size << "\n\n";
      return max_kernel_work_group_size;
    }
    // otherwise, the work-size will stay the originally desired one
    return work_group_size;
  }
}

int main() {
  /* 2 ^ 20 = 262144, large enough and is not going to cause memory allocation
   * problems in vector as opposed to going with the absoulte maximum power of 2
   * number of elements we could store in the vector. Moreover, it is safe to go
   * with this size in case of running on host device */
  constexpr size_t iterations = 1 << 20;
  // 2 ^ 10 = 1024, desirable number of work-items per work-group
  size_t work_group_size = 1 << 10;

  auto selector = sycl::default_selector{};
  std::cout << "SYCL runtime using " << typeid(selector).name() << "\n";

  // Container for the sum calculated per each work-group.
  std::vector<sycl::cl_int> results;

  // Generate random points on the host - one point for each thread
  std::vector<sycl::float2> points;
  points.reserve(iterations);
  // fill up with random values
  std::random_device r;
  std::default_random_engine e(r());
  std::uniform_real_distribution<float> dist;
  for (size_t i = 0; i < iterations; i++) {
    points[i].x() = dist(e);
    points[i].y() = dist(e);
  }

  try {
    // create a SYCL queue with a handler for asynchronous errors
    sycl::queue queue(selector, [](sycl::exception_list errors) {
      for (auto error : errors) {
        try {
          std::rethrow_exception(error);
        } catch (sycl::exception e) {
          std::cerr << "There is an exception in the kernel\n";
          std::cerr << e.what() << "\n";
        }
      }
    });

    // Get device and display information: name and platform
    auto device = queue.get_device();
    std::cout << "Selected " << device.get_info<sycl::info::device::name>()
              << " on platform "
              << device.get_info<sycl::info::device::platform>()
                     .get_info<sycl::info::platform::name>()
              << "\n";

    // Define the SYCL program and kernel
    auto context = queue.get_context();
    sycl::program program(context);
    // Build the Monte-Carlo Pi kernel
    program.build_with_kernel_type<monte_carlo_pi_kernel>();
    sycl::kernel kernel = program.get_kernel<monte_carlo_pi_kernel>();

    /* If the desired work-group size doesn't satisfy the device, define a
     * perfect/max work-group depending on the selected device and kernel
     * maximum size allowance */
    work_group_size = get_best_work_group_size(work_group_size, device, kernel);

    /* Size of the total sums that are going to be stored in the results vector
     * is set based on the defined work-group size */
    results.resize(iterations / work_group_size);

    // Allocate device memory
    sycl::buffer<sycl::cl_float2> points_buf(points.data(),
                                             sycl::range<1>(iterations));
    sycl::buffer<sycl::cl_int> results_buf(
        results.data(), sycl::range<1>(iterations / work_group_size));

    queue.submit([&](sycl::handler& cgh) {
      const size_t global_size = iterations;
      const size_t local_size = work_group_size;

      // Get access to the data (points and results) on the device
      auto points_ptr =
          points_buf.get_access<sycl::access::mode::read,
                                sycl::access::target::constant_buffer>(cgh);
      auto results_ptr = results_buf.get_access<sycl::access::mode::write>(cgh);
      // Allocate local memory on the device (to compute results)
      sycl::accessor<sycl::cl_int, 1, sycl::access::mode::write,
                     sycl::access::target::local>
          local_results_ptr(sycl::range<1>(local_size), cgh);

      // Run the kernel
      cgh.parallel_for(
          program.get_kernel<monte_carlo_pi_kernel>(),
          sycl::nd_range<1>(sycl::range<1>(global_size),
                            sycl::range<1>(local_size)),
          monte_carlo_pi_kernel(points_ptr, results_ptr, local_results_ptr));
    });
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception caught: " << e.what();
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "C++ exception caught: " << e.what();
    return 2;
  }

  // Sum the results (auto copied back to host)
  int in_circle = 0;
  for (auto& result : results) {
    in_circle += result;
  }

  // Calculate the final result of "pi"
  float pi =
      (4.0f * static_cast<float>(in_circle)) / static_cast<float>(iterations);
  std::cout << "pi = " << pi << "\n";

  return 0;
}
