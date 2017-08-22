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
 *  tuple_utils.hpp
 *
 *  Description:
 *    C++11-compatible utilities for manipulating tuples.
 *
 **************************************************************************/

#pragma once

#include <tuple>


/* Given a function name and a one-line body, defines an auto-return-type
 * function with these parameters. */
#define AUTO_FUNC(signature, body) \
  auto signature->decltype(body) { return body; }

// Utilities for indexing tuples
template <size_t...>
struct index_sequence {
  using type = index_sequence;
};

namespace {
template <class S1, class S2>
struct concat;

template <size_t... I1, size_t... I2>
struct concat<index_sequence<I1...>, index_sequence<I2...>>
    : index_sequence<I1..., (sizeof...(I1) + I2)...> {};

template <class S1, class S2>
using Concat = typename concat<S1, S2>::type;

template <size_t N>
struct gen_seq;
}

template <size_t N>
using make_index_sequence = typename gen_seq<N>::type;

namespace {
template <size_t N>
struct gen_seq
    : Concat<make_index_sequence<N / 2>, make_index_sequence<N - N / 2>> {};

template <>
struct gen_seq<0> : index_sequence<> {};
template <>
struct gen_seq<1> : index_sequence<0> {};
}

template <typename... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

// Sets variadic reference arguments to the values in the tuple.
namespace {
template <size_t K, typename tuple_t, typename Arg>
void setvar_helper(tuple_t const& tpl, Arg& arg) {
  static_assert(K == std::tuple_size<tuple_t>() - 1, "");
  arg = std::get<K>(tpl);
}

template <size_t K, typename tuple_t, typename Arg, typename... Args>
void setvar_helper(tuple_t const& tpl, Arg& arg, Args&... args) {
  arg = std::get<K>(tpl);
  setvar_helper<K + 1>(tpl, args...);
}
}

template <typename tuple_t, typename... Args>
void setvar(tuple_t const& tpl, Args&... args) {
  setvar_helper<0>(tpl, args...);
}

// Calls function with argument tuple.
namespace {
template <typename Func, typename Tuple, size_t... Ids>
AUTO_FUNC(call_help(Func&& func, Tuple&& args, index_sequence<Ids...>),
          func(std::get<Ids>(std::forward<Tuple>(args))...))
}

template <
    typename Func, typename Tuple,
    size_t Size = std::tuple_size<typename std::decay<Tuple>::type>::value,
    typename Ids = make_index_sequence<Size>>
AUTO_FUNC(call(Func&& func, Tuple&& args),
          call_help(std::forward<Func>(func), std::forward<Tuple>(args), Ids{}))

// Cuts off elements from the beginning and end of a tuple to make it shorter.
namespace {
  template <size_t Off, size_t... Ids>
  index_sequence<(Ids + Off)...> offset_index_sequence(index_sequence<Ids...>);

  template <size_t From, size_t N>
  using make_index_range =
      decltype(offset_index_sequence<From>(make_index_sequence<N>{}));

  template <typename Tuple, size_t... Ids>
  auto squash_tuple_help(Tuple && tpl, index_sequence<Ids...>)
      ->decltype(std::make_tuple(std::get<Ids>(std::forward<Tuple>(tpl))...)) {
    return std::make_tuple(std::get<Ids>(std::forward<Tuple>(tpl))...);
  }
}

// Cuts off OffB elements from the beginning of a tuple and OffE elements from
// the end.
template <
    size_t OffB, size_t OffE, typename Tuple,
    size_t Size = std::tuple_size<typename std::decay<Tuple>::type>::value,
    typename Ids = make_index_range<OffB, Size - OffB - OffE>>
auto squash_tuple(Tuple&& tpl)
    -> decltype(squash_tuple_help(std::forward<Tuple>(tpl), Ids{})) {
  return squash_tuple_help(std::forward<Tuple>(tpl), Ids{});
}

// Adds tuples by calling operator+ elementwise.
namespace {
template <typename TupleA, typename TupleB, size_t... Ids>
auto add_tuples_help(TupleA&& a, TupleB&& b, index_sequence<Ids...>)
    -> decltype(std::make_tuple(std::get<Ids>(std::forward<TupleA>(a)) +
                                std::get<Ids>(std::forward<TupleB>(b))...)) {
  return std::make_tuple(std::get<Ids>(std::forward<TupleA>(a)) +
                         std::get<Ids>(std::forward<TupleB>(b))...);
}
}

template <
    typename TupleA, typename TupleB,
    size_t SizeA = std::tuple_size<typename std::decay<TupleA>::type>::value,
    size_t SizeB = std::tuple_size<typename std::decay<TupleB>::type>::value,
    typename Ids = make_index_sequence<SizeA>>
auto add_tuples(TupleA&& a, TupleB&& b)
    -> decltype(add_tuples_help(std::forward<TupleA>(a),
                                std::forward<TupleB>(b), Ids{})) {
  static_assert(SizeA == SizeB, "Cannot add tuples of differing sizes.");
  return add_tuples_help(std::forward<TupleA>(a), std::forward<TupleB>(b),
                         Ids{});
}

// Multiplies a tuple by a value, elementwise.
namespace {
template <typename Tuple, typename Mult, size_t... Ids>
AUTO_FUNC(mult_tuple_help(Tuple&& a, Mult&& m, index_sequence<Ids...>),
          std::make_tuple(std::get<Ids>(std::forward<Tuple>(a)) *
                          std::forward<Mult>(m)...))
}

template <
    typename Tuple, typename Mult,
    size_t Size = std::tuple_size<typename std::decay<Tuple>::type>::value,
    typename Ids = make_index_sequence<Size>>
AUTO_FUNC(mult_tuple(Tuple&& a, Mult&& m),
          mult_tuple_help(std::forward<Tuple>(a), std::forward<Mult>(m), Ids{}))

/* Returns a tuple of the specified types, all elements initialized with the
 * given
 * value of type T. Example: a tuple of vectors of different types, but all of
 * the
 * same size. */
template <typename T, typename... Vals>
AUTO_FUNC(make_tuple_multi(T val), std::make_tuple(Vals(val)...))

// Returns the first argument and ignores the second
template <typename T, typename U>
AUTO_FUNC(passthrough(T&& t, U&& u), std::forward<T>(t))

// Makes a tuple of N elements of the same type T.
template <typename T, size_t... Ids>
AUTO_FUNC(make_homogenous_tuple_help(T val, index_sequence<Ids...>),
          std::make_tuple(passthrough(val, Ids)...))

template <typename T, size_t N>
AUTO_FUNC(make_homogenous_tuple(T val),
          make_homogenous_tuple_help(val, make_index_sequence<N>{}))

/* Transforms a tuple by executing the provided function on every element
 * of the tuple. */
template <typename Tuple, typename Func, size_t... Ids>
AUTO_FUNC(transform_tuple_help(Tuple&& tpl, Func, index_sequence<Ids...>),
          std::make_tuple((Func{}.template operator()(
              std::get<Ids>(std::forward<Tuple>(tpl))))...))

template <typename Tuple, typename Func>
AUTO_FUNC(transform_tuple(Tuple&& tpl, Func),
          transform_tuple_help(
              std::forward<Tuple>(tpl), Func{},
              make_index_sequence<
                  std::tuple_size<typename std::decay<Tuple>::type>::value>{}))

// Zips two tuples into a tuple of pairs.
template <typename TupleA, typename TupleB, size_t... Ids>
AUTO_FUNC(
    zip_tuples_help(TupleA&& a, TupleB&& b, index_sequence<Ids...>),
    std::make_tuple(std::make_pair(std::get<Ids>(std::forward<TupleA>(a)),
                                   std::get<Ids>(std::forward<TupleB>(b)))...))

template <typename TupleA, typename TupleB,
          typename Idx = make_index_sequence<
              std::tuple_size<typename std::decay<TupleA>::type>::value>>
AUTO_FUNC(zip_tuples(TupleA&& a, TupleB&& b),
          zip_tuples_help(std::forward<TupleA>(a), std::forward<TupleB>(b),
                          Idx{}))
