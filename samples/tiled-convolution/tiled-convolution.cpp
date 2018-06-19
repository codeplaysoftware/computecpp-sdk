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
 *    Sample code that illustrates how to divide data into tiles and lauch
 *    separate kerenl per tile by using range accessors and parallel for with
 *    offset in SYCL.
 *
 **************************************************************************/
#include <SYCL/sycl.hpp>
#include <chrono>
#include <ctime>
#include <iostream>

// configuring the openCL local size
template <typename index_t>
struct opencl_configuration_t {
  static constexpr index_t local_size_m = 1;
  static constexpr index_t local_size_n = 32;
};

// define the 2-d matrix size
template <typename index_t>
class matrix_size_t {
 public:
  const index_t m;
  const index_t n;
  matrix_size_t(const index_t m_, const index_t n_) : m(m_), n(n_) {}
  inline index_t size() const { return (m * n); }
};

// the tiled based convolution functor
template <typename read_accessor_t, typename write_accessor_t, typename index_t,
          typename mat_size_t>
class conv {
 private:
  read_accessor_t fil_acc;
  read_accessor_t in_acc;
  write_accessor_t out_acc;
  const mat_size_t total_size;
  const mat_size_t fil_size;

 public:
  // constructing the functor
  conv(read_accessor_t fil_acc_, read_accessor_t in_acc_,
       write_accessor_t out_acc_, const mat_size_t total_size_,
       const mat_size_t fil_size_)
      : fil_acc(fil_acc_),
        in_acc(in_acc_),
        out_acc(out_acc_),
        total_size(total_size_),
        fil_size(fil_size_) {}
  void inline operator()(cl::sycl::nd_item<2> item_id) {
    index_t id_m = item_id.get_global(0);
    index_t id_n = item_id.get_global(1);
    index_t m, f_m;
    index_t n, f_n;
    typename write_accessor_t::value_type val = 0.0;
#ifdef __SYCL_DEVICE_ONLY__
#pragma umroll
#endif
    for (f_m = 0, m = -1; f_m < 3; f_m++, m++) {
      index_t in_id_m = (id_m + m >= 0) ? id_m + m : 0;
      in_id_m = (in_id_m < total_size.m) ? in_id_m : total_size.m - 1;
#ifdef __SYCL_DEVICE_ONLY__
#pragma umroll
#endif
      for (f_n = 0, n = -1; f_n < 3; f_n++, n++) {
        index_t in_id_n = (id_n + n >= 0) ? id_n + n : 0;
        in_id_n = (in_id_n < total_size.n) ? in_id_n : total_size.n - 1;
        val += (in_acc[in_id_m][in_id_n] * fil_acc[f_m][f_n]);
      }
    }
    out_acc[id_m][id_n] = val / fil_size.size();
  }
};

// round up function to construct the global and local size
template <typename index_t>
inline index_t round_up(const index_t x, const index_t y) {
  return ((((x) + (y) -1) / (y)) * (y));
}

// calculating halo around each tile
template <typename index_t>
void inline compute_index(const index_t total_size_dim,
                          const index_t mat_size_dim,
                          const index_t fil_size_dim,
                          const index_t tile_offset_dim, index_t& range_src_dim,
                          index_t& offset_src_dim) {
  if (tile_offset_dim == 0 && mat_size_dim < total_size_dim) {
    offset_src_dim = tile_offset_dim;
    range_src_dim = mat_size_dim + (fil_size_dim / 2);
  } else if (tile_offset_dim != 0 &&
             (tile_offset_dim + mat_size_dim) < total_size_dim) {
    offset_src_dim = tile_offset_dim - (fil_size_dim / 2);
    range_src_dim = mat_size_dim + fil_size_dim - 1;
  } else if (tile_offset_dim != 0 &&
             (tile_offset_dim + mat_size_dim) >= total_size_dim) {
    offset_src_dim = tile_offset_dim - (fil_size_dim / 2);
    range_src_dim = mat_size_dim + (fil_size_dim / 2);
  } else if (tile_offset_dim == 0 && mat_size_dim >= total_size_dim) {
    offset_src_dim = tile_offset_dim;
    range_src_dim = mat_size_dim;
  }
}

int main() {
  using data_t = float;
  using index_t = int;
  // sycl read type
  static constexpr auto read_t = cl::sycl::access::mode::read;
  // sycl write type
  static constexpr auto write_t = cl::sycl::access::mode::write;
  // sycl global buffer type
  static constexpr auto global_buffer_t =
      cl::sycl::access::target::global_buffer;
  // read_accessor type
  using read_accessor_t =
      cl::sycl::accessor<data_t, 2, read_t, global_buffer_t>;
  // write accessor type
  using write_accessor_t =
      cl::sycl::accessor<data_t, 2, write_t, global_buffer_t>;
  // opencl configuration type
  using ocl_config_t = opencl_configuration_t<index_t>;

  // total input data size
  auto total_buffer = matrix_size_t<index_t>{1024, 1024};
  // tile size per iteration
  auto mat_size = matrix_size_t<index_t>{512, 512};
  auto fil_size = matrix_size_t<index_t>{3, 3};

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

  auto property_list =
      cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
  // constructing a SYCL queue for CVengine OpenCL device where
  // automatically build the underlying context and command_queue for the
  // chosen device.
  auto sycl_queue = cl::sycl::queue(
      (cl::sycl::default_selector()),
      [&](cl::sycl::exception_list l) {
        bool error = false;
        for (auto e : l) {
          try {
            std::rethrow_exception(e);
          } catch (const cl::sycl::exception& e) {
            auto clError = e.get_cl_code();
            std::cout << e.what() << "CL ERROR CODE : " << clError << std::endl;
            error = true;
          } catch (...) {
            std::cerr << " error  \n";
            std::abort();
          }
        }
        if (error) {
          throw std::runtime_error("SYCL errors detected");
        }
      },
      property_list);

  // input SYCL buffer
  auto in_buff = cl::sycl::buffer<data_t, 2>(
      input.data(), cl::sycl::range<2>(total_buffer.m, total_buffer.n));
  // mask(filter) SYCL buffer
  auto fil_buff = cl::sycl::buffer<data_t, 2>(
      filter.data(), cl::sycl::range<2>(fil_size.m, fil_size.n));
  // output SYCL buffer
  auto out_buff = cl::sycl::buffer<data_t, 2>(
      cl::sycl::range<2>(total_buffer.m, total_buffer.n));
  double total_submission_time = 0;
  double total_execution_time = 0;
  double avarage_submission_time = 0;
  double avarage_execution_time = 0;
  double avarage_application_execution_time = 0;
  double total_application_execution_time = 0;
  index_t host_offset_m = 0;
  // launching tiled-based kernel via two nested for-loop
  for (index_t m = 0; m < num_host_tile_m; m++) {
    index_t host_offset_n = 0;
    for (index_t n = 0; n < num_host_tile_n; n++) {
      index_t range_src_m, offset_src_m;
      index_t range_src_n, offset_src_n;
      // calculating the halo for first dimension of the tile
      compute_index(total_buffer.m, mat_size.m, fil_size.m, host_offset_m,
                    range_src_m, offset_src_m);
      // calculating the halo for the second dimension of the tile
      compute_index(total_buffer.n, mat_size.n, fil_size.n, host_offset_n,
                    range_src_n, offset_src_n);
      auto start = std::chrono::system_clock::now();
      // events[n + m * num_host_tile_n] =
      auto event = sycl_queue.submit([&](cl::sycl::handler& cgh) {
        // filter
        auto fil_acc = fil_buff.get_access<read_t, global_buffer_t>(cgh);
        // input
        auto in_acc = in_buff.get_access<read_t, global_buffer_t>(
            cgh, cl::sycl::range<2>(range_src_m, range_src_n),
            cl::sycl::id<2>(offset_src_m, offset_src_n));
        // output
        auto out_acc = out_buff.get_access<write_t, global_buffer_t>(
            cgh, cl::sycl::range<2>(mat_size.m, mat_size.n),
            cl::sycl::id<2>(host_offset_m, host_offset_n));
        // global size m
        auto global_size_m = round_up(mat_size.m, ocl_config_t::local_size_m);
        // global size n
        auto global_size_n = round_up(mat_size.n, ocl_config_t::local_size_n);
        // constructing the kernel
        cgh.parallel_for(cl::sycl::nd_range<2>(
                             cl::sycl::range<2>(global_size_m, global_size_n),
                             cl::sycl::range<2>(ocl_config_t::local_size_m,
                                                ocl_config_t::local_size_n),
                             cl::sycl::id<2>(host_offset_m, host_offset_n)),
                         conv<read_accessor_t, write_accessor_t, index_t,
                              matrix_size_t<index_t>>(fil_acc, in_acc, out_acc,
                                                      total_buffer, fil_size));
      });
      // this part is used to profile each tiled kernel
#ifdef PROFILE_SYCL
      auto i = n + m * num_host_tile_n;
      event.wait();
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double, std::milli>
          current_application_execution_time = end - start;
      total_application_execution_time +=
          current_application_execution_time.count();
      auto submit_time = event.get_profiling_info<
          cl::sycl::info::event_profiling::command_submit>();
      auto start_time = event.get_profiling_info<
          cl::sycl::info::event_profiling::command_start>();
      auto end_time = event.get_profiling_info<
          cl::sycl::info::event_profiling::command_end>();
      auto current_submission_time = (start_time - submit_time) / 1000000.0f;
      auto current_execution_time = (end_time - start_time) / 1000000.0f;

      total_execution_time += current_execution_time;
      total_submission_time += current_submission_time;
#ifdef PER_EVENT_PROFILING
      std::cout << "event, " << i << " , current_kernel_submission_time, "
                << current_submission_time
                << " , current_kernel_execution_time, "
                << current_execution_time
                << ", current_application_execution_time, "
                << current_application_execution_time.count() << "\n";
#endif
#endif
      host_offset_n += mat_size.n;
    }
    host_offset_m += mat_size.m;
  }
  // this part is used to profile the total execution time
#ifdef PROFILE_SYCL
  avarage_submission_time =
      total_submission_time / float(num_host_tile_n * num_host_tile_m);
  avarage_execution_time =
      total_execution_time / float(num_host_tile_n * num_host_tile_m);
  avarage_application_execution_time = total_application_execution_time /
                                       float(num_host_tile_n * num_host_tile_m);
  std::cout << " local_size_0, " << ocl_config_t::local_size_m
            << " , local_size_1, " << ocl_config_t::local_size_n
            << " , total_kernel_submission_time, " << total_submission_time
            << " , total_kernel_execution_time, " << total_execution_time
            << " , avarage_kernel_submission_time, " << avarage_submission_time
            << " , avarage_kernel_execution_time, " << avarage_execution_time
            << " , total_application_execution_time, "
            << total_application_execution_time
            << " , avarage_application_execution_time, "
            << avarage_application_execution_time << "\n";
#endif
  // check the correctness of the result
  auto out_data = out_buff.get_access<read_t>();
  for (index_t m = 0; m < total_buffer.m; m++)
    for (index_t n = 0; n < total_buffer.m; n++)
      if (!(std::abs(index_t(out_data[m][n] - (input_data * filter_data))) <
            1e-4))
        std::cout << " The result is wrong " << std::endl;
  std::cout << " The result is correct " << std::endl;
  return 0;
}
