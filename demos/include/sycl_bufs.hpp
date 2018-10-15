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
 *  sycl_bufs.hpp
 *
 *  Description:
 *    A class for storing and accessing buffers of values of any given types.
 *
 **************************************************************************/

#pragma once

#include <type_traits>

#include <CL/sycl.hpp>

#include "tuple_utils.hpp"


// Template function object which transforms buffers to device read accessors
struct BufToReadAccFunc {
  template <typename In>
  AUTO_FUNC(
      // pair of (buffer, handler)
      operator()(In&& in),
      std::forward<In>(in)
          .first.template get_access<cl::sycl::access::mode::read>(
              *std::forward<In>(in).second))
};

// Template function object which transforms buffers to device write accessors
struct BufToDcdWriteAccFunc {
  // pair of (buffer, handler)
  template <typename In>
  AUTO_FUNC(operator()(In&& in),
            std::forward<In>(in)
                .first.template get_access<cl::sycl::access::mode::discard_write>(
                    *std::forward<In>(in).second))
};

// Template function object which transforms buffers to host read accessors
struct BufToHostReadAccFunc {
  template <typename In>
  auto operator()(In&& in) -> decltype(
      std::forward<In>(in)
          .template get_access<cl::sycl::access::mode::read>()) {
    return std::forward<In>(in)
        .template get_access<cl::sycl::access::mode::read>();
  }
};

// Template function object which transforms buffers to host write accessors
struct BufToHostDcdWriteAccFunc {
  template <typename In>
  auto operator()(In&& in) -> decltype(
      std::forward<In>(in)
          .template get_access<cl::sycl::access::mode::discard_write>()) {
    return std::forward<In>(in)
        .template get_access<cl::sycl::access::mode::discard_write>();
  }
};

// Which buffers to read
template <size_t... Ids>
struct read_bufs_t {};

// Which buffers to write
template <size_t... Ids>
struct write_bufs_t {};

// Provides a buffer for elements of each of the variadic types Ts
template <typename... Ts>
class SyclBufs {
  std::tuple<cl::sycl::buffer<Ts, 1>...> m_bufs;

 public:
  SyclBufs(size_t N)
      : m_bufs(make_tuple_multi<size_t, cl::sycl::buffer<Ts, 1>...>(N)) {}

  // Returns a tuple of read accessors for the selected buffers
  template <size_t... Ids>
  AUTO_FUNC(
      gen_read_accs(cl::sycl::handler& cgh, read_bufs_t<Ids...>),
      transform_tuple(zip_tuples(std::make_tuple(std::get<Ids>(m_bufs)...),
                                 make_homogenous_tuple<cl::sycl::handler*,
                                                       sizeof...(Ids)>(&cgh)),
                      BufToReadAccFunc{}))

  // Returns a tuple of write accessors for the selected buffers
  template <size_t... Ids>
  AUTO_FUNC(
      gen_write_accs(cl::sycl::handler& cgh, write_bufs_t<Ids...>),
      transform_tuple(zip_tuples(std::make_tuple(std::get<Ids>(m_bufs)...),
                                 make_homogenous_tuple<cl::sycl::handler*,
                                                       sizeof...(Ids)>(&cgh)),
                      BufToDcdWriteAccFunc{}))

  // Returns a tuple of host read accessors for the selected buffers
  template <size_t... Ids>
  AUTO_FUNC(gen_host_read_accs(read_bufs_t<Ids...>),
            transform_tuple(std::make_tuple(std::get<Ids>(m_bufs)...),
                            BufToHostReadAccFunc{}))

  // Returns a tuple of host write accessors for the selected buffers
  template <size_t... Ids>
  AUTO_FUNC(gen_host_write_accs(write_bufs_t<Ids...>),
            transform_tuple(std::make_tuple(std::get<Ids>(m_bufs)...),
                            BufToHostDcdWriteAccFunc{}))
};
