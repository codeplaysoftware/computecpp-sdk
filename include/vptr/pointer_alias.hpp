/***************************************************************************
 *
 *  Copyright (C) 2017 Codeplay Software Limited
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
 *  pointer_alias.hpp
 *
 *  Description:
 *    Alias functions that obtain a pointer of the given type from an
 *    accessor.
 *
 * Authors:
 *
 *    Ruyman Reyes   Codeplay Software Ltd.
 *    Mehdi Goli     Codeplay Software Ltd.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#ifndef CL_SYCL_POINTER_ALIAS
#define CL_SYCL_POINTER_ALIAS

namespace cl {
namespace sycl {
namespace codeplay {

template <typename T, typename AccessorT>
typename cl::sycl::global_ptr<T>::pointer_t get_device_ptr_as(AccessorT& acc) {
  return reinterpret_cast<typename cl::sycl::global_ptr<T>::pointer_t>(
      acc.get_pointer().get());
}

template <typename T, typename AccessorT>
T* get_host_ptr_as(AccessorT& acc) {
  return reinterpret_cast<T*>(acc.get_pointer());
}

}  // namespace codeplay
}  // namespace sycl
}  // namespace cl

#endif  // CL_SYCL_POINTER_ALIAS
