/***************************************************************************
 *
 *  Copyright (C) 2018 Codeplay Software Limited
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
 *  tiled-convolution.cpp
 *
 *  Description:
 *    Sample code that illustrates how to divide data into tiles and launch
 *    separate kernels per tile by using ranged accessors and parallel for with
 *    offsets in SYCL. See the readme for further information
 *
 **************************************************************************/
#ifndef COMMON_HPP
#define COMMON_HPP
#include <SYCL/sycl.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>

// define the 2-d matrix size
class matrix_size_t {
 public:
  const int m;
  const int n;
  matrix_size_t(const int m_, const int n_) : m(m_), n(n_) {}
  inline int size() const { return (m * n); }
};

struct opencl_configuration_t {
  // best case 32
  static constexpr int cache_line = 32;
  // best case :(mat_size.n) 1024
  static constexpr int col_per_thread = 1024;
  // bast case : (matsize.m / global_work_size) 32
  static constexpr int row_per_tread = 32;
  static constexpr int work_group_reduction_factor = 2;  // best case 2
  static constexpr int row_per_work_item = 2;
};

// round up function to construct the global and local size
inline int round_up(const int x, const int y) {
  return ((x + y - 1) / y) * y;
}

template <bool>
inline bool do_check(bool cond) {
  return cond;
}
template <>
inline bool do_check<false>(bool) {
  return true;
}

void inline profiler(
    std::vector<cl::sycl::event>& events,
    const std::vector<std::chrono::time_point<std::chrono::system_clock>>&
        starts) {
  double total_execution_time = 0;
  double per_tile_execution_time = 0;
  double per_tile_application_execution_time = 0;
  double total_application_execution_time = 0;
  int size = events.size();
  for (int i = 0; i < size; i++) {
    events[i].wait();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli>
        current_application_execution_time = end - starts[i];
    total_application_execution_time +=
        current_application_execution_time.count();
    auto start_time = events[i]
                          .template get_profiling_info<
                              cl::sycl::info::event_profiling::command_start>();
    auto end_time = events[i]
                        .template get_profiling_info<
                            cl::sycl::info::event_profiling::command_end>();
    auto current_execution_time = (end_time - start_time) / 1000000.0f;

    total_execution_time += current_execution_time;
#ifdef PER_EVENT_PROFILING
    std::cout << "Tile, " << i << " , current_kernel_execution_time(ms), "
              << current_execution_time
              << ", current_application_execution_time(ms), "
              << current_application_execution_time.count() << "\n";
#endif
  }
  // this part is used to profile the total execution time
  per_tile_execution_time = total_execution_time / double(size);
  per_tile_application_execution_time =
      total_application_execution_time / double(size);
  std::cout << "  total_kernel_execution_time(ms), " << total_execution_time
            << " , total_application_execution_time(ms), "
            << total_application_execution_time
            << " , avarage_kernel_execution_time(ms), "
            << per_tile_execution_time
            << " , avarage_application_execution_time(ms), "
            << per_tile_application_execution_time << "\n";
}

template <typename mat_size_t, typename host_accessor_t, typename data_t>
int validate(const mat_size_t dims, host_accessor_t host_acc,
             const data_t ref_data) {
  for (int m = 0; m < dims.m; m++) {
    for (int n = 0; n < dims.n; n++) {
      if (!(std::fabs(data_t(host_acc[m][n] - ref_data)) < 1e-4)) {
        std::cout << " The result is wrong " << std::endl;
        std::cout << "m : " << m << ", n : " << n << ", host_acc[m][n] "
                  << host_acc[m][n] << "\n";
        return -1;
      }
    }
  }
  std::cout << " The result is correct " << std::endl;
  return 0;
}
#endif  // COMMON_HPP