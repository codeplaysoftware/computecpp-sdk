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
#ifndef COPY_HPP
#define COPY_HPP
#include "common.hpp"

template <typename read_accessor_t, typename write_accessor_t,
          typename mat_size_t>
class copy_from_rectangular_kernel {
 private:
  read_accessor_t in_acc;    // input accessor
  write_accessor_t out_acc;  // output accessor
  const mat_size_t size;     //  input size

 public:
  copy_from_rectangular_kernel(read_accessor_t in_acc_,
                               write_accessor_t out_acc_,
                               const mat_size_t size_)
      : in_acc(in_acc_), out_acc(out_acc_), size(size_) {}
  void inline operator()(cl::sycl::nd_item<2> item_id) {
    int in_m = item_id.get_global_id(0);
    int in_n = item_id.get_global_id(1);
    int out_m = in_m - item_id.get_offset()[0];
    int out_n = in_n - item_id.get_offset()[1];
    if ((out_m >= size.m) || (out_n >= size.n))
      return;
    out_acc[out_m][out_n] = in_acc[in_m][in_n];
  }
};

template <typename read_accessor_t, typename write_accessor_t,
          typename mat_size_t>
class copy_to_rectangular_kernel {
 private:
  read_accessor_t in_acc;    // input accessor
  write_accessor_t out_acc;  // output accessor
  const mat_size_t size;     //  input size

 public:
  copy_to_rectangular_kernel(read_accessor_t in_acc_, write_accessor_t out_acc_,
                             const mat_size_t size_)
      : in_acc(in_acc_), out_acc(out_acc_), size(size_) {}
  void inline operator()(cl::sycl::nd_item<2> item_id) {
    int out_m = item_id.get_global_id(0);
    int out_n = item_id.get_global_id(1);
    int in_m = out_m - item_id.get_offset()[0];
    int in_n = out_n - item_id.get_offset()[1];
    if ((in_m >= size.m) || (in_n >= size.n))
      return;
    out_acc[out_m][out_n] = in_acc[in_m][in_n];
  }
};

template <typename kernel_t, typename read_buff_t, typename write_buff_t,
          typename mat_size_t>
void inline copy_rectangular(cl::sycl::queue& sycl_queue,
                             cl::sycl::program& sycl_program,
                             read_buff_t src_buf, write_buff_t dst_buf,
                             mat_size_t range_size, mat_size_t offset) {
  sycl_queue.submit([&](cl::sycl::handler& cgh) {
    // this must be constant buffer
    auto dst_acc =
        dst_buf.template get_access<cl::sycl::access::mode::write,
                                    cl::sycl::access::target::global_buffer>(
            cgh);
    auto src_acc =
        src_buf.template get_access<cl::sycl::access::mode::read,
                                    cl::sycl::access::target::global_buffer>(
            cgh);
    int work_group_size =
        sycl_queue.get_device()
            .get_info<cl::sycl::info::device::max_work_group_size>();
    int global_size_m = round_up(range_size.m, 1);
    int global_size_n = round_up(range_size.n, work_group_size);
    cgh.parallel_for(
        sycl_program.template get_kernel<kernel_t>(),
        cl::sycl::nd_range<2>(cl::sycl::range<2>(global_size_m, global_size_n),
                              cl::sycl::range<2>(1, work_group_size),
                              cl::sycl::id<2>(offset.m, offset.n)),
        kernel_t(src_acc, dst_acc, range_size));
  });
  sycl_queue.wait();
};

#endif  // COPY_HPP