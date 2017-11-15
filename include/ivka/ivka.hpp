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
 *  ivka.hpp
 *
 *  Description:
 *    Implementation of a type trait the indicates whether a type can be
 *    used as a kernel argument or not.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <type_traits>

/* Clang identifies itself as GCC, but we really want to be sure that
 * we're disabling GCC 4.X and earlier */
#if !defined(__clang__) && defined(__GNUC__) && (__GNUC__ < 5)
#error "is_valid_kernel_arg<T> requires GCC >= 5"
#endif

/* Least specialised template, most things aren't accessors */
template <typename>
struct is_accessor {
  static constexpr auto value = false;
};

/* More specialised: must match accessor template args exactly, and then
 * uses std::is_same to make sure that the thing passed in is the same
 * type as an accessor with the same template args. */
template <template <typename, int, cl::sycl::access::mode,
                    cl::sycl::access::target, cl::sycl::access::placeholder>
          class T, typename U, int d, cl::sycl::access::mode m,
          cl::sycl::access::target t, cl::sycl::access::placeholder p>
struct is_accessor<T<U, d, m, t, p>> {
  static constexpr auto value =
      std::is_same<T<U, d, m, t, p>, cl::sycl::accessor<U, d, m, t, p>>::value;
};

/* Least specialised, chosen for most typenames. Matches the spec definition
 * of things that can be kernel arguments. */
template <typename T, typename U = void>
struct is_valid_kernel_arg {
  static constexpr auto value =
      std::is_standard_layout<T>::value && std::is_trivially_copyable<T>::value;
};

/* More specialised, chosen when T is an instance of an accessor. All
 * accessors are valid kernel arguments. */
template <typename T>
struct is_valid_kernel_arg<
    T, typename std::enable_if<is_accessor<T>::value>::type> {
  static constexpr auto value = true;
};
