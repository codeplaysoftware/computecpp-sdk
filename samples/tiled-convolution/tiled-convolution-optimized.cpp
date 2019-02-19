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
 *  tiled-convolution-optimized.cpp
 *
 *  Description:
 *    Sample code that illustrates how to divide data into tiles and launch
 *    separate kernels per tile by using ranged accessors and parallel for with
 *    offsets in SYCL. Optimized version for embedded devices.
 *    See the README for further information.
 *
 **************************************************************************/
#include "common.hpp"
#include "tiled-conv.hpp"

#include <SYCL/codeplay.hpp>

int main() {
  using data_t = input_data_info::data_t;

  // total input data size
  constexpr auto total_buffer =
      matrix_size_t{input_data_info::N, input_data_info::N};
  // tile size per iteration
  constexpr auto mat_size = total_buffer / input_data_info::divider;
  constexpr auto fil_size = matrix_size_t{3, 3};

  // constructing the tile size
  auto num_host_tile_n = total_buffer.n / mat_size.n;
  auto num_host_tile_m = total_buffer.m / mat_size.m;
  // input value
  auto input_data = data_t(0.6);
  // mask filter value
  auto filter_data = data_t(0.3);
  // input array
  std::vector<data_t> input(total_buffer.size(), input_data);
  // mask array
  std::vector<data_t> filter(fil_size.size(), filter_data);

  // enabling SYCL queue profiling
  auto prop_list =
      cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};

  auto sycl_queue = cl::sycl::queue(
      [&](cl::sycl::exception_list l) {
        bool error = false;
        for (auto e : l) {
          try {
            std::rethrow_exception(e);
          } catch (const cl::sycl::exception& e) {
            auto clError = e.get_cl_code();
            std::cout << e.what() << "CL ERROR CODE : " << clError << std::endl;
            error = true;
          }
        }
        if (error) {
          throw std::runtime_error("SYCL errors detected");
        }
      },
      prop_list);

  const auto context_bound_property =
      cl::sycl::property::buffer::context_bound(sycl_queue.get_context());

  // building kernel before the execution by using program class
  // This will reduce the program overhead
  // input SYCL buffer
  auto in_buff = cl::sycl::buffer<data_t, 2>(
      input.data(), cl::sycl::range<2>(total_buffer.m, total_buffer.n),
      {context_bound_property});
  in_buff.set_write_back(false);
  // mask(filter) SYCL buffer
  auto fill_buff = cl::sycl::buffer<data_t, 2>(
      filter.data(), cl::sycl::range<2>(fil_size.m, fil_size.n),
      {context_bound_property});
  fill_buff.set_write_back(false);
  // output SYCL buffer
  auto out_buff = cl::sycl::buffer<data_t, 2>(
      cl::sycl::range<2>(total_buffer.m, total_buffer.n),
      {context_bound_property});
  static constexpr auto read_t = cl::sycl::access::mode::read;
  static constexpr auto write_t = cl::sycl::access::mode::write;
  static constexpr auto global_buffer_t =
      cl::sycl::access::target::global_buffer;
  using read_accessor_t =
      cl::sycl::accessor<data_t, 2, read_t, global_buffer_t>;
  using write_accessor_t =
      cl::sycl::accessor<data_t, 2, write_t, global_buffer_t>;

  using conv_kernel_type = conv<read_accessor_t, write_accessor_t>;
  // building kernel before the execution by using program class
  // This will reduce the program overhead
  auto sycl_program = cl::sycl::program(sycl_queue.get_context());
  sycl_program.build_with_kernel_type<conv_kernel_type>();
  // launching tiled-based kernel via two nested for-loop
  int host_offset_m = 0;
  std::vector<cl::sycl::event> events(num_host_tile_m * num_host_tile_n);
  std::vector<std::chrono::time_point<std::chrono::system_clock>> starts(
      num_host_tile_m * num_host_tile_n);

  const auto use_onchip_memory_property =
      cl::sycl::codeplay::property::buffer::use_onchip_memory(
          cl::sycl::codeplay::property::prefer);
  // Create temporary input buffer using on-chip memory
  auto temp_in_buff = cl::sycl::buffer<data_t, 2>(
      cl::sycl::range<2>(mat_size.m + 2, mat_size.n + 2),
      {context_bound_property, use_onchip_memory_property});
  // Create temporary output buffer using on-chip memory
  auto temp_out_buff = cl::sycl::buffer<data_t, 2>(
      cl::sycl::range<2>(mat_size.m, mat_size.n),
      {context_bound_property, use_onchip_memory_property});

  // Force output to zero
  sycl_queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = out_buff.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(out_buff.get_range(), init_to_zero<decltype(acc)>{acc});
  });

  for (int m = 0; m < num_host_tile_m; m++) {
    int host_offset_n = 0;
    for (int n = 0; n < num_host_tile_n; n++) {
      int i = n + m * num_host_tile_n;
      int range_src_m, offset_src_m;
      int range_src_n, offset_src_n;
      std::array<bool, 2> clamped_edge_m = {};
      std::array<bool, 2> clamped_edge_n = {};

      // calculating the halo for first dimension of the tile
      compute_index(total_buffer.m, mat_size.m, fil_size.m, host_offset_m,
                    range_src_m, offset_src_m, clamped_edge_m);
      // calculating the halo for the second dimension of the tile
      compute_index(total_buffer.n, mat_size.n, fil_size.n, host_offset_n,
                    range_src_n, offset_src_n, clamped_edge_n);

      // copying a specific region form in_buffer to the
      // temporary input buffer
      sycl_queue.submit([&](cl::sycl::handler& cgh) {
        auto temp_in_acc = temp_in_buff.template get_access<write_t>(cgh);
        auto in_acc = in_buff.get_access<read_t>(
            cgh, cl::sycl::range<2>(range_src_m, range_src_n),
            cl::sycl::id<2>(offset_src_m, offset_src_n));
        cgh.copy(in_acc, temp_in_acc);
      });

      // execute the tile convolution
      tiled_cov<conv_kernel_type>(
          sycl_queue, sycl_program, temp_in_buff, fill_buff, temp_out_buff,
          mat_size, matrix_size_t{range_src_m, range_src_n}, fil_size, i,
          events, starts, clamped_edge_m[0], clamped_edge_n[0]);

      // copy the data back
      sycl_queue.submit([&](cl::sycl::handler& cgh) {
        auto temp_out_acc = temp_out_buff.template get_access<read_t>(cgh);
        auto out_acc = out_buff.get_access<write_t>(
            cgh, cl::sycl::range<2>(mat_size.m, mat_size.n),
            cl::sycl::id<2>(host_offset_m, host_offset_n));
        cgh.copy(temp_out_acc, out_acc);
      });

      host_offset_n += mat_size.n;
    }
    host_offset_m += mat_size.m;
  }
  profiler(events, starts);

  const auto result = validate(total_buffer, out_buff.get_access<read_t>(),
                               (filter_data * input_data));
  return result ? 0 : -1;
}
