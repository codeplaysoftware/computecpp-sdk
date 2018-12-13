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
 *  common.hpp
 *
 *  Description:
 *    Common code for different implementations of the tiled convolution sample.
 *
 **************************************************************************/
#ifndef COMMON_HPP
#define COMMON_HPP

#include <CL/sycl.hpp>
#include <SYCL/codeplay.hpp>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>

// define the 2-d matrix size
class matrix_size_t {
 public:
  const int m;
  const int n;
  constexpr int size() const { return m * n; }
  constexpr matrix_size_t operator/(int divider) const {
    return {m / divider, n / divider};
  }
};

struct opencl_configuration_t {
  static constexpr int cache_line = 4;
  // best case :(mat_size.n) 1024
  static constexpr int col_per_thread = 1024;
  static constexpr int row_per_thread = 4;
  static constexpr int work_group_reduction_factor = 2;  // best case 2
  static constexpr int row_per_work_item = 1;
  static constexpr matrix_size_t local_size = {1, 32};
};

struct input_data_info {
  using data_t = float;
  static constexpr int N = 512;
  static constexpr int divider = 2;
};

// round up function to construct the global and local size
constexpr int round_up(const int x, const int y) {
  return ((x + y - 1) / y) * y;
}

constexpr matrix_size_t round_up(const matrix_size_t x, const matrix_size_t y) {
  return {round_up(x.m, y.m), round_up(x.n, y.n)};
}

namespace {

template <bool>
inline bool do_check(bool cond) {
  return cond;
}
template <>
inline bool do_check<false>(bool) {
  return true;
}

inline void profiler(
    std::vector<cl::sycl::event>& events,
    const std::vector<std::chrono::time_point<std::chrono::system_clock>>&
        starts) {
  constexpr auto cmd_start_info =
      cl::sycl::info::event_profiling::command_start;
  constexpr auto cmd_end_info = cl::sycl::info::event_profiling::command_end;
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
    auto start_time = events[i].get_profiling_info<cmd_start_info>();
    auto end_time = events[i].get_profiling_info<cmd_end_info>();
    auto current_execution_time = (end_time - start_time) / 1000000.0f;

    total_execution_time += current_execution_time;
    std::cout << "Tile, " << i << " , current_kernel_execution_time(ms), "
              << current_execution_time
              << ", current_application_execution_time(ms), "
              << current_application_execution_time.count() << "\n";
  }
  // this part is used to profile the total execution time
  per_tile_execution_time = total_execution_time / size;
  per_tile_application_execution_time = total_application_execution_time / size;
  std::cout << "  total_kernel_execution_time(ms), " << total_execution_time
            << " , total_application_execution_time(ms), "
            << total_application_execution_time
            << " , average_kernel_execution_time(ms), "
            << per_tile_execution_time
            << " , average_application_execution_time(ms), "
            << per_tile_application_execution_time << "\n";
}

template <typename host_accessor_t, typename data_t>
bool validate(const matrix_size_t dims, host_accessor_t host_acc,
              const data_t ref_data) {
  for (int m = 0; m < dims.m; m++) {
    for (int n = 0; n < dims.n; n++) {
      if (!(std::fabs(host_acc[m][n] - ref_data) < 1e-4)) {
        std::cout << " The result is wrong\n";
        std::cout << "m : " << m << ", n : " << n << ", host_acc[m][n] "
                  << host_acc[m][n] << "\n";
        return false;
      }
    }
  }
  std::cout << " The result is correct\n";
  return true;
}

// calculating halo around each tile
inline void compute_index(const int total_size_dim, const int mat_size_dim,
                          const int fil_size_dim, const int tile_offset_dim,
                          int& range_src_dim, int& offset_src_dim,
                          std::array<bool, 2>& clamp_edge_dim) {
  // Clamp to left/top
  clamp_edge_dim[0] = tile_offset_dim == 0;
  offset_src_dim =
      tile_offset_dim - (clamp_edge_dim[0] ? 0 : (fil_size_dim / 2));

  range_src_dim = mat_size_dim;

  if (clamp_edge_dim[0] && mat_size_dim < total_size_dim) {
    range_src_dim += (fil_size_dim / 2);
    // Don't clamp to right/left
    clamp_edge_dim[1] = false;
  } else if (!clamp_edge_dim[0] &&
             (tile_offset_dim + mat_size_dim) < total_size_dim) {
    range_src_dim += fil_size_dim - 1;
    // Don't clamp to right/left
    clamp_edge_dim[1] = false;
  } else if (!clamp_edge_dim[0] &&
             (tile_offset_dim + mat_size_dim) >= total_size_dim) {
    range_src_dim += (fil_size_dim / 2);
    // clamp to right/left
    clamp_edge_dim[1] = true;
  } else if (clamp_edge_dim[0] && mat_size_dim >= total_size_dim) {
    range_src_dim += 0;
    // clamp to right/left
    clamp_edge_dim[1] = true;
  }
}

template <typename accessor_t>
struct init_to_zero {
  accessor_t acc;
  void operator()(cl::sycl::id<2> i) const { acc[i] = 0; }
};

}  // namespace

#endif  // COMMON_HPP
