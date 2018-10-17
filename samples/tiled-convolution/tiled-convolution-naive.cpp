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
#include "common.hpp"
#include "copy.hpp"

// calculating halo around each tile
void inline compute_index(const int total_size_dim, const int mat_size_dim,
                          const int fil_size_dim, const int tile_offset_dim,
                          int& range_src_dim, int& offset_src_dim,
                          std::array<bool, 2>& clamp_edge_dim) {
  if (tile_offset_dim == 0 && mat_size_dim < total_size_dim) {
    offset_src_dim = tile_offset_dim;
    range_src_dim = mat_size_dim + (fil_size_dim / 2);
    // clamp to left/top
    clamp_edge_dim[0] = true;
    // dont clamp to right/left
    clamp_edge_dim[1] = false;
  } else if (tile_offset_dim != 0 &&
             (tile_offset_dim + mat_size_dim) < total_size_dim) {
    offset_src_dim = tile_offset_dim - (fil_size_dim / 2);
    range_src_dim = mat_size_dim + fil_size_dim - 1;
    // dont clamp to left/top
    clamp_edge_dim[0] = false;
    // dont clamp to right/left
    clamp_edge_dim[1] = false;
  } else if (tile_offset_dim != 0 &&
             (tile_offset_dim + mat_size_dim) >= total_size_dim) {
    offset_src_dim = tile_offset_dim - (fil_size_dim / 2);
    range_src_dim = mat_size_dim + (fil_size_dim / 2);
    // dont clamp to left/top
    clamp_edge_dim[0] = false;
    // clamp to right/left
    clamp_edge_dim[1] = true;
  } else if (tile_offset_dim == 0 && mat_size_dim >= total_size_dim) {
    offset_src_dim = tile_offset_dim;
    range_src_dim = mat_size_dim;
    // clamp to left/top
    clamp_edge_dim[0] = true;
    // clamp to right/left
    clamp_edge_dim[1] = true;
  }
}

// the tiled based convolution functor
template <typename read_accessor_t, typename write_accessor_t,
          typename mat_size_t>
class conv {
 private:
  read_accessor_t fil_acc;      // filter accessor
  read_accessor_t in_acc;       // input accessor
  write_accessor_t out_acc;     // output accessor
  const mat_size_t total_size;  // total input size
  const mat_size_t fil_size;    // filter size
  const std::array<bool, 2> clamp_edge_m;
  const std::array<bool, 2> clamp_edge_n;

 public:
  conv(read_accessor_t fil_acc_, read_accessor_t in_acc_,
       write_accessor_t out_acc_, const mat_size_t total_size_,
       const mat_size_t fil_size_, const std::array<bool, 2> clamp_edge_m_,
       const std::array<bool, 2> clamp_edge_n_)
      : fil_acc(fil_acc_),
        in_acc(in_acc_),
        out_acc(out_acc_),
        total_size(total_size_),
        fil_size(fil_size_),
        clamp_edge_m(clamp_edge_m_),
        clamp_edge_n(clamp_edge_n_) {}
  void inline operator()(cl::sycl::nd_item<2> item_id) {
    int id_m = item_id.get_global_id(0);
    int id_n = item_id.get_global_id(1);
    int m, fil_m, n, fil_n;
    typename write_accessor_t::value_type val = 0.0;
    const int m_start_offset = clamp_edge_m[0] ? 0 : fil_size.m / 2;
    const int m_end_offset = clamp_edge_m[1] ? 0 : fil_size.m / 2;
    const int n_start_offset = clamp_edge_n[0] ? 0 : fil_size.n / 2;
    const int n_end_offset = clamp_edge_n[1] ? 0 : fil_size.n / 2;
    int m_out = total_size.m - m_start_offset - m_end_offset;
    int n_out = total_size.n - n_start_offset - n_end_offset;
    // disabling all out of bound threads
    if ((id_m >= m_out) || (id_n >= n_out))
      return;

    id_m += m_start_offset;
    id_n += n_start_offset;

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
    out_acc[item_id.get_global_id(0)][item_id.get_global_id(1)] =
        val / fil_size.size();
  }
};

template <typename kernel_t, typename read_buff_t, typename write_buff_t,
          typename mat_size_t>
void inline tiled_cov(
    cl::sycl::queue& sycl_queue, cl::sycl::program& sycl_program,
    read_buff_t in_buff, read_buff_t fill_buff, write_buff_t out_buff,
    mat_size_t out_range_size, mat_size_t in_range_size,
    mat_size_t fil_range_size, int i, std::vector<cl::sycl::event>& events,
    std::vector<std::chrono::time_point<std::chrono::system_clock>>& starts,
    std::array<bool, 2>& clamped_edge_m, std::array<bool, 2>& clamped_edge_n) {
  // execute the tile
  starts[i] = std::chrono::system_clock::now();
  events[i] = sycl_queue.submit([&](cl::sycl::handler& cgh) {
    // this must be constant buffer
    auto in_acc =
        in_buff.template get_access<cl::sycl::access::mode::read,
                                    cl::sycl::access::target::global_buffer>(
            cgh);
    auto fil_acc =
        fill_buff.template get_access<cl::sycl::access::mode::read,
                                      cl::sycl::access::target::global_buffer>(
            cgh);
    auto out_acc =
        out_buff.template get_access<cl::sycl::access::mode::write,
                                     cl::sycl::access::target::global_buffer>(
            cgh);
    auto global_size_m = round_up(out_range_size.m, 1);
    auto global_size_n = round_up(out_range_size.n, 32);
    cgh.parallel_for(
        sycl_program.template get_kernel<kernel_t>(),
        cl::sycl::nd_range<2>(cl::sycl::range<2>(global_size_m, global_size_n),
                              cl::sycl::range<2>(1, 32)),
        kernel_t(fil_acc, in_acc, out_acc, in_range_size, fil_range_size,
                 clamped_edge_m, clamped_edge_n));
  });
}

int main() {
  using data_t = float;
  // total input data size
  auto total_buffer = matrix_size_t{1024, 1024};
  // tile size per iteration
  auto mat_size = matrix_size_t{1024, 1024};
  auto fil_size = matrix_size_t{3, 3};

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
  auto property_list =
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
      property_list);
  // building kernel before the execution by using program class
  // This will reduce the program overhead
  // input SYCL buffer
  auto in_buff = cl::sycl::buffer<data_t, 2>(
      input.data(), cl::sycl::range<2>(total_buffer.m, total_buffer.n),
      {cl::sycl::property::buffer::context_bound(sycl_queue.get_context())});
  // mask(filter) SYCL buffer
  auto fill_buff = cl::sycl::buffer<data_t, 2>(
      filter.data(), cl::sycl::range<2>(fil_size.m, fil_size.n),
      {cl::sycl::property::buffer::context_bound(sycl_queue.get_context())});
  // output SYCL buffer
  auto out_buff = cl::sycl::buffer<data_t, 2>(
      cl::sycl::range<2>(total_buffer.m, total_buffer.n),
      {cl::sycl::property::buffer::context_bound(sycl_queue.get_context())});
  static constexpr auto read_t = cl::sycl::access::mode::read;
  static constexpr auto write_t = cl::sycl::access::mode::write;
  static constexpr auto global_buffer_t =
      cl::sycl::access::target::global_buffer;
  using read_accessor_t =
      cl::sycl::accessor<data_t, 2, read_t, global_buffer_t>;
  using write_accessor_t =
      cl::sycl::accessor<data_t, 2, write_t, global_buffer_t>;
  using from_kernel_type =
      copy_from_rectangular_kernel<read_accessor_t, write_accessor_t,
                                   matrix_size_t>;
  using to_kernel_type =
      copy_to_rectangular_kernel<read_accessor_t, write_accessor_t,
                                 matrix_size_t>;

  using conv_kernel_type =
      conv<read_accessor_t, write_accessor_t, matrix_size_t>;
  // building kernel before the execution by using program class
  // This will reduce the program overhead
  auto sycl_program = cl::sycl::program(sycl_queue.get_context());
  sycl_program.build_with_kernel_type<from_kernel_type>();
  sycl_program.build_with_kernel_type<to_kernel_type>();
  sycl_program.build_with_kernel_type<conv_kernel_type>();
  // launching tiled-based kernel via two nested for-loop
  int host_offset_m = 0;
  std::vector<cl::sycl::event> events(num_host_tile_m * num_host_tile_n);
  std::vector<std::chrono::time_point<std::chrono::system_clock>> starts(
      num_host_tile_m * num_host_tile_n);
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
      // the temporary input buffer value
      auto temp_in_buff = cl::sycl::buffer<data_t, 2>(
          cl::sycl::range<2>(range_src_m, range_src_n),
          {cl::sycl::property::buffer::context_bound(
              sycl_queue.get_context())});
      auto temp_out_buff = cl::sycl::buffer<data_t, 2>(
          cl::sycl::range<2>(mat_size.m, mat_size.n),
          {cl::sycl::property::buffer::context_bound(
              sycl_queue.get_context())});

      // copying an specific
      // region form in_buffer to the temporary input buffer
      /* sycl_queue.submit([&](cl::sycl::handler& cgh) {
         auto temp_in_acc =
             temp_in_buff.template get_access<write_t, global_buffer_t>(cgh);
         auto in_acc = in_buff.get_access<read_t, global_buffer_t>(
             cgh, cl::sycl::range<2>(range_src_m, range_src_n),
             cl::sycl::id<2>(offset_src_m, offset_src_n));
         cgh.copy(in_acc, temp_in_acc);
       });*/
      // temporary work around rectangular copy in SYCL
      copy_rectangular<from_kernel_type>(
          sycl_queue, sycl_program, in_buff, temp_in_buff,
          matrix_size_t(range_src_m, range_src_n),
          matrix_size_t(offset_src_m, offset_src_n));

      // execute the tile convolution
      tiled_cov<conv_kernel_type>(
          sycl_queue, sycl_program, temp_in_buff, fill_buff, temp_out_buff,
          mat_size, matrix_size_t(range_src_m, range_src_n), fil_size, i,
          events, starts, clamped_edge_m, clamped_edge_n);

      // copy the data back
      /*sycl_queue.submit([&](cl::sycl::handler& cgh) {
        auto temp_out_acc = temp_out_buff.template get_access<read_t>(cgh);
        auto out_acc = out_buff.get_access<write_t, global_buffer_t>(
            cgh, cl::sycl::range<2>(mat_size.m, mat_size.n),
            cl::sycl::id<2>(host_offset_m, host_offset_n));
        cgh.copy(temp_out_acc, out_acc);
      });*/
      // temporary work around rectangular copy in SYCL
      copy_rectangular<to_kernel_type>(
          sycl_queue, sycl_program, temp_out_buff, out_buff, mat_size,
          matrix_size_t(host_offset_m, host_offset_n));

      host_offset_n += mat_size.n;
    }
    host_offset_m += mat_size.m;
  }
  profiler(events, starts);

  validate(total_buffer, out_buff.get_access<read_t>(),
           (filter_data * input_data));
  return 0;
}