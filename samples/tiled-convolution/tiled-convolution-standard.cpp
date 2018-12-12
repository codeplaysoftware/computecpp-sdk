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
 *  tiled-convolution-standard.cpp
 *
 *  Description:
 *    Sample code that illustrates how to divide data into tiles and launch
 *    separate kernels per tile by using ranged accessors and parallel for with
 *    offsets in SYCL.
 *    See the README for further information.
 *
 **************************************************************************/
#include "common.hpp"

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>

// the tiled based convolution functor
template <typename read_accessor_t, typename write_accessor_t>
class conv {
 private:
  read_accessor_t fil_acc;         // filter accessor
  read_accessor_t in_acc;          // input accessor
  write_accessor_t out_acc;        // output accessor
  const matrix_size_t total_size;  // total input size
  const matrix_size_t fil_size;    // filter size

 public:
  conv(read_accessor_t fil_acc_, read_accessor_t in_acc_,
       write_accessor_t out_acc_, const matrix_size_t total_size_,
       const matrix_size_t fil_size_)
      : fil_acc(fil_acc_),
        in_acc(in_acc_),
        out_acc(out_acc_),
        total_size(total_size_),
        fil_size(fil_size_) {}
  void inline operator()(cl::sycl::nd_item<2> item_id) {
    int id_m = item_id.get_global_id(0);  // global id with offset m
    int id_n = item_id.get_global_id(1);  // global id with offset n
    // input index m , filter index fil_m , input index n, filter index fil_n
    int m, fil_m, n, fil_n;
    typename write_accessor_t::value_type val = 0.0;
    // loop over filter size m and add halo to input m
    for (fil_m = 0, m = -1; fil_m < fil_size.m; fil_m++, m++) {
      int in_id_m = (id_m + m >= 0) ? id_m + m : 0;
      in_id_m = (in_id_m < total_size.m) ? in_id_m : total_size.m - 1;
      // loop over filter size n and add halo to input n
      for (fil_n = 0, n = -1; fil_n < fil_size.n; fil_n++, n++) {
        int in_id_n = (id_n + n >= 0) ? id_n + n : 0;
        in_id_n = (in_id_n < total_size.n) ? in_id_n : total_size.n - 1;
        val += (in_acc[in_id_m][in_id_n] * fil_acc[fil_m][fil_n]);
      }
    }
    out_acc[id_m][id_n] = val / fil_size.size();
  }
};

int main() {
  using data_t = input_data_info::data_t;

  // total input data size
  constexpr auto total_buffer =
      matrix_size_t{input_data_info::N, input_data_info::N};
  // tile size per iteration
  constexpr auto mat_size = total_buffer / input_data_info::divider;
  constexpr auto fil_size = matrix_size_t{3, 3};

  static constexpr auto read_t = cl::sycl::access::mode::read;
  static constexpr auto write_t = cl::sycl::access::mode::write;
  static constexpr auto global_buffer_t =
      cl::sycl::access::target::global_buffer;
  using read_accessor_t =
      cl::sycl::accessor<data_t, 2, read_t, global_buffer_t>;
  using write_accessor_t =
      cl::sycl::accessor<data_t, 2, write_t, global_buffer_t>;

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

  using kernel_type = conv<read_accessor_t, write_accessor_t>;
  // building kernel before the execution by using program class
  // This will reduce the program overhead
  auto sycl_program = cl::sycl::program(sycl_queue.get_context());
  sycl_program.build_with_kernel_type<kernel_type>();
  // input SYCL buffer
  auto in_buff = cl::sycl::buffer<data_t, 2>(
      input.data(), cl::sycl::range<2>(total_buffer.m, total_buffer.n));
  // mask(filter) SYCL buffer
  auto fil_buff = cl::sycl::buffer<data_t, 2>(
      filter.data(), cl::sycl::range<2>(fil_size.m, fil_size.n));
  // output SYCL buffer
  auto out_buff = cl::sycl::buffer<data_t, 2>(
      cl::sycl::range<2>(total_buffer.m, total_buffer.n));

  // Force output to zero
  sycl_queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = out_buff.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for(out_buff.get_range(), init_to_zero<decltype(acc)>{acc});
  });

  int host_offset_m = 0;
  std::vector<cl::sycl::event> events(num_host_tile_m * num_host_tile_n);
  std::vector<std::chrono::time_point<std::chrono::system_clock>> starts(
      num_host_tile_m * num_host_tile_n);
  // launching tiled-based kernel via two nested for-loop
  for (int m = 0; m < num_host_tile_m; m++) {
    int host_offset_n = 0;
    for (int n = 0; n < num_host_tile_n; n++) {
      int range_src_m, offset_src_m;
      int range_src_n, offset_src_n;
      std::array<bool, 2> clamped_edge = {};
      // calculating the halo for first dimension of the tile
      compute_index(total_buffer.m, mat_size.m, fil_size.m, host_offset_m,
                    range_src_m, offset_src_m, clamped_edge);
      // calculating the halo for the second dimension of the tile
      compute_index(total_buffer.n, mat_size.n, fil_size.n, host_offset_n,
                    range_src_n, offset_src_n, clamped_edge);
      int i = n + m * num_host_tile_n;
      starts[i] = std::chrono::system_clock::now();
      events[i] = sycl_queue.submit([&](cl::sycl::handler& cgh) {
        auto fil_acc = fil_buff.get_access<read_t, global_buffer_t>(cgh);
        auto in_acc = in_buff.get_access<read_t, global_buffer_t>(
            cgh, cl::sycl::range<2>(range_src_m, range_src_n),
            cl::sycl::id<2>(offset_src_m, offset_src_n));
        auto out_acc = out_buff.get_access<write_t, global_buffer_t>(
            cgh, cl::sycl::range<2>(mat_size.m, mat_size.n),
            cl::sycl::id<2>(host_offset_m, host_offset_n));
        constexpr auto global_size =
            round_up(mat_size, opencl_configuration_t::local_size);
        cgh.parallel_for(
            sycl_program.template get_kernel<kernel_type>(),
            cl::sycl::nd_range<2>(
                cl::sycl::range<2>(global_size.m, global_size.n),
                cl::sycl::range<2>(opencl_configuration_t::local_size.m,
                                   opencl_configuration_t::local_size.n),
                cl::sycl::id<2>(host_offset_m, host_offset_n)),
            kernel_type(fil_acc, in_acc, out_acc, total_buffer, fil_size));
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
