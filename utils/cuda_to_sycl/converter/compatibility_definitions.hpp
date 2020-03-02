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
 *  compatibility_definitions.hpp
 *
 *  Description:
 *    Alias functions that obtain a pointer of the given type from an
 *    accessor.
 *
 * Authors:
 *
 *    Mehdi Goli     Codeplay Software Ltd.
 *    Ruyman Reyes   Codeplay Software Ltd.
 *
 **************************************************************************/
/** compatibility definitions **/
#ifndef COMPATIBILITY_DEFINITIONS_HPP
#define COMPATIBILITY_DEFINITIONS_HPP
// General SYCL headers
#include <CL/sycl.hpp>

// SDK Virtual pointer
#include <stl-tuple/STLTuple.hpp>
#include <vptr/virtual_ptr.hpp>

#define __global__
#define __device__

struct dim3 {
  int x, y, z;
  using self_t = dim3;
  inline self_t operator*(self_t otherdim3) const {
    return self_t(otherdim3.x * x, otherdim3.y * y, otherdim3.z * z);
  }
  dim3() {}
  dim3(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
};

namespace cl {
namespace sycl {
namespace codeplay {
PointerMapper& get_global_pointer_mapper() {
  static PointerMapper globalMapper_s;
  return globalMapper_s;
}

/**
 * Memcpy Direction Configuration
 */
enum Kind : int {
  HostToHost = 0,     /**< Host   -> Host */
  HostToDevice = 1,   /**< Host   -> Device */
  DeviceToHost = 2,   /**< Device -> Host */
  DeviceToDevice = 3, /**< Device -> Device */
  Default = 4 /**< Direction of the transfer is inferred from the pointer
                 values. Requires unified virtual addressing */
};

template <typename T1, typename T2, Kind kind_t>
struct copy_t;

template <typename T1, typename T2>
struct copy_t<T1, T2, cl::sycl::codeplay::Kind::HostToDevice> {
  static cl::sycl::event sycl_copy_conversion(cl::sycl::queue dQ, T1* src,
                                              T2* dst, bool async) {
    auto event = dQ.submit([&](cl::sycl::handler& h) {
      auto acc_ = get_global_pointer_mapper().get_access(dst, h);
      h.copy((uint8_t*)src, acc_);
    });
    if (!async) {
      event.wait();
    }
    return event;
  }
};

template <typename T1, typename T2>
struct copy_t<T1, T2, cl::sycl::codeplay::Kind::DeviceToHost> {
  static cl::sycl::event sycl_copy_conversion(cl::sycl::queue dQ, T1* src,
                                              T2* dst, bool async) {
    auto event = dQ.submit([&](cl::sycl::handler& h) {
      auto acc_ = get_global_pointer_mapper().get_access(src, h);
      h.copy(acc_, (uint8_t*)dst);
    });
    if (!async) {
      event.wait();
    }
    return event;
  }
};

/** cuda_copy_conversion.
 * @brief Following the same API ordering as cudaMemCpy , converts the
 * direct memcpy into a SYCL copy operation, matching directionality.
 *
 * @todo Pointer offsets not implemented.
 * @todo Select correct directionality and trigger the right access mode.
 * @note Type dispatch mechanism can be used to implement the selection of the
 * appropriate copy functor.
 *
 * @param output Pointer to the destination
 */
template <Kind kind_t, typename T1, typename T2>
cl::sycl::event cuda_copy_conversion(cl::sycl::queue dQ, T1* src, T2* dst,
                                     size_t, bool async = false) {
  return copy_t<T1, T2, kind_t>::sycl_copy_conversion(dQ, src, dst, async);
}

template <typename T>
static cl::sycl::event sycl_memset(cl::sycl::queue dQ, T* dst, int value,
                                   size_t, bool async = false) {
  auto event = dQ.submit([&](cl::sycl::handler& h) {
    auto acc = get_global_pointer_mapper().get_access(dst, h);
    // The cast to uint8_t is here to match the behaviour of the standard
    // memset.
    h.fill(acc, (static_cast<uint8_t>(value)));
  });
  if (!async) {
    event.wait();
  }
  return event;
}

template <typename T>
using acc_t =
    accessor<T, 1, access::mode::read_write, access::target::global_buffer>;
using local_acc_t =
    accessor<uint8_t, 1, access::mode::read_write, access::target::local>;

// This class is used to preserve the original type of the data
template <typename T1, typename T2>
struct real_accessor_t;
template <typename T1, typename T2>
struct real_accessor_t<T1, acc_t<T2>> {
  using real_t = std::remove_pointer<T1>;
  acc_t<T2> acc_;
  real_accessor_t(acc_t<T2> acc) : acc_(acc) {}
};

template <typename conv_t>
struct converter {
  using type = conv_t;
  static conv_t inline convert(conv_t c, cl::sycl::handler&) { return c; }
};

template <typename T>
struct converter<T*> {
  using type = real_accessor_t<T, acc_t<uint8_t>>;
  static type inline convert(T* vir_ptr, cl::sycl::handler& h) {
    return type(get_global_pointer_mapper().get_access(vir_ptr, h));
  }
};

template <typename functor_t>
struct kernel_dispatcher;
template <typename blockIdx_t, typename threadIdx_t, typename blockDim_t,
          typename gridDim_t, typename nd_item_t, typename... Param_t,
          template <class...> class user_kernel_t>
struct kernel_dispatcher<user_kernel_t<blockIdx_t, threadIdx_t, blockDim_t,
                                       gridDim_t, nd_item_t, Param_t...>> {
  using type = user_kernel_t<blockIdx_t, threadIdx_t, blockDim_t, gridDim_t,
                             nd_item_t, Param_t...>;
  utility::tuple::Tuple<Param_t...> t;
  blockIdx_t blockIdx;
  threadIdx_t threadIdx;
  blockDim_t blockDim;
  gridDim_t gridDim;

  kernel_dispatcher(Param_t... param)
      : t(utility::tuple::make_tuple(param...)){};
  void operator()(nd_item_t it_) {
    blockIdx = blockIdx_t(it_.get_group(0), it_.get_group(1), it_.get_group(2));
    threadIdx = threadIdx_t(it_.get_local_id(0), it_.get_local_id(1),
                            it_.get_local_id(2));
    blockDim = blockDim_t(it_.get_local_range(0), it_.get_local_range(1),
                          it_.get_local_range(2));
    gridDim = gridDim_t(it_.get_group_range(0), it_.get_group_range(1),
                        it_.get_group_range(2));

    caller(it_, utility::tuple::IndexRange<0, sizeof...(Param_t)>());
  }

  template <size_t... Is>
  void caller(cl::sycl::nd_item<3> it, utility::tuple::IndexList<Is...>) {
    auto kernel_functor = type(blockIdx, threadIdx, blockDim, gridDim, it,
                               utility::tuple::get<Is>(t)...);
    kernel_functor.wrapper(
        utility::tuple::IndexRange<0, sizeof...(Param_t) - 1>());
  }
};

template <typename kernel>
struct CudaCommandGroup;
template <typename... Param_t, template <class...> class KernelT>
class CudaCommandGroup<KernelT<Param_t...>> {
 private:
  dim3 gridSize_;
  dim3 blockSize_;
  int local_mem_size_;
  // kernel parameters
  utility::tuple::Tuple<Param_t...> t;

 public:
  CudaCommandGroup(dim3 gridSize, dim3 blockSize, int local_mem_size,
                   Param_t... param)
      : gridSize_{gridSize},
        blockSize_{blockSize},
        local_mem_size_{local_mem_size},
        t{utility::tuple::make_tuple(param...)} {}
  CudaCommandGroup(dim3 gridSize, dim3 blockSize, Param_t... param)
      : CudaCommandGroup(gridSize, blockSize, sizeof(void*), param...) {}
  CudaCommandGroup(int gridSize, int blockSize, int local_mem_size,
                   Param_t... param)
      : CudaCommandGroup(dim3(gridSize, 1, 1), dim3(blockSize, 1, 1),
                         local_mem_size, param...) {}
  CudaCommandGroup(int gridSize, int blockSize, Param_t... param)
      : CudaCommandGroup(gridSize, blockSize, sizeof(void*), param...) {}
  void operator()(cl::sycl::handler& h) {
    auto t2 = utility::tuple::append(
        t, utility::tuple::make_tuple(local_acc_t(local_mem_size_, h)));
    caller(h, t2, utility::tuple::IndexRange<0, sizeof...(Param_t) + 1>());
  }
  template <typename handler_t, typename... append_param_t, size_t... Is>
  void caller(handler_t& h, utility::tuple::Tuple<append_param_t...> t2,
              utility::tuple::IndexList<Is...>) {
    auto globalrange = (gridSize_ * blockSize_);
    using u_ker_t = KernelT<dim3, dim3, dim3, dim3, cl::sycl::nd_item<3>,
                            typename converter<append_param_t>::type...>;
    using kernel_t = kernel_dispatcher<u_ker_t>;
    auto func = kernel_t((
        converter<append_param_t>::convert(utility::tuple::get<Is>(t2), h))...);
    h.parallel_for(
        nd_range<3>{
            cl::sycl::range<3>(globalrange.x, globalrange.y, globalrange.z),
            cl::sycl::range<3>(blockSize_.x, blockSize_.y, blockSize_.z)},
        func);
  }
};

// raw pointer class is used to extract the multi-pointer from accessor class
template <typename type_t>
struct raw_pointer {
  using type = type_t;
  static inline type get_pointer(type_t dt) { return dt; }
};
template <typename T1, typename T2>
struct raw_pointer<real_accessor_t<T1, acc_t<T2>>> {
  using raw_type = typename cl::sycl::multi_ptr<
      T1, cl::sycl::access::address_space::global_space>::pointer_t;
  using type =
      cl::sycl::multi_ptr<T1, cl::sycl::access::address_space::global_space>;
  static inline type get_pointer(real_accessor_t<T1, acc_t<T2>> dt) {
    return type(reinterpret_cast<T1*>(dt.acc_.get_pointer().get()));
  }
};

template <>
struct raw_pointer<local_acc_t> {
  using type = typename cl::sycl::multi_ptr<
      uint8_t, cl::sycl::access::address_space::local_space>::pointer_t;
  static inline type get_pointer(local_acc_t dt) {
    return dt.get_pointer().get();
  }
};

// The generic caller function
template <typename F, typename... Args>
void call_func(F& func, Args... args) {
  func.__execute__(args...);
}

// Generic kernel functor to construct and execute the kernel
template <typename u_k_t>
class Generic_Kernel_Functor;

template <typename blockIdx_t, typename threadIdx_t, typename blockDim_t,
          typename gridDim_t, typename nd_item_t, typename... Param_t,
          template <class...> class u_k_t>
class Generic_Kernel_Functor<u_k_t<blockIdx_t, threadIdx_t, blockDim_t,
                                   gridDim_t, nd_item_t, Param_t...>> {
 public:
  using user_kernel_type = u_k_t<blockIdx_t, threadIdx_t, blockDim_t, gridDim_t,
                                 nd_item_t, Param_t...>;
  blockIdx_t blockIdx;
  threadIdx_t threadIdx;
  blockDim_t blockDim;
  gridDim_t gridDim;
  nd_item_t it_;
  // N+1 elements. The first N elements are the parameters of the cuda kernel
  // the last element is the local memory
  utility::tuple::Tuple<Param_t...> t;
  Generic_Kernel_Functor(blockIdx_t blockIdx_, threadIdx_t threadIdx_,
                         blockDim_t blockDim_, gridDim_t gridDim_, nd_item_t it,
                         Param_t... param)
      : blockIdx(blockIdx_),
        threadIdx(threadIdx_),
        blockDim(blockDim_),
        gridDim(gridDim_),
        it_(it),
        t(utility::tuple::make_tuple(param...)) {}

  void __syncthreads() {
    it_.barrier(cl::sycl::access::fence_space::global_and_local);
  }

  template <typename T>
  T* get_local_mem() {
    return cl::sycl::multi_ptr<T, cl::sycl::access::address_space::local_space>(
        (reinterpret_cast<T*>(raw_pointer<local_acc_t>::get_pointer(
            utility::tuple::get<sizeof...(Param_t) - 1>(t)))));
  }

  template <typename... removed_local_param_t, size_t... Is>
  inline void unpack(utility::tuple::Tuple<removed_local_param_t...> t_removed,
                     utility::tuple::IndexList<Is...>) {
    call_func(*(reinterpret_cast<user_kernel_type*>(this)),
              (raw_pointer<removed_local_param_t>::get_pointer(
                  utility::tuple::get<Is>(t_removed)))...);
  }
  template <size_t... Is>
  inline void wrapper(utility::tuple::IndexList<Is...> indices) {
    unpack(utility::tuple::make_tuple((utility::tuple::get<Is>(t))...),
           indices);
  }
};

}  // namespace codeplay
}  // namespace sycl
}  // namespace cl
#endif