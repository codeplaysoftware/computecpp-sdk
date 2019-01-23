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
 *  integrator.hpp
 *
 *  Description:
 *    Provides generic and typesafe PDE integration methods.
 *
 **************************************************************************/

#pragma once

#include "tuple_utils.hpp"

/* Given a function `func` expressing a derivative of order N, a time step size
 * `step`,
 * and N+1 values `vals` with the initial conditions of y(N-1), .., y1, y, t, in
 * that order,
 * returns the new values of y(N-1), ..., y1, y, t after the step in a tuple.
 * Uses Runge-Kutta RK4 integration. */
template <typename time_t, typename func_t, typename... Args>
std::tuple<Args...> integrate_step_rk4(func_t func, time_t step, Args... vals) {
  static_assert(sizeof...(Args) >= 2,
                "Do you want infinite loops in your compiler? Because this is "
                "how you get infinite loops in your compiler.");

  // preserve the initial values
  auto const init = std::make_tuple(vals...);
  // arguments to the function describing the derivative: yN = f(y(N-1), .., y1,
  // y, t)
  // start with initial values
  auto args = init;
  // no addition to args here, since it's the first step

  // k0s: yN, .., y1, 1.0
  // these values will be added to args to advance them for a half step
  // 1.0 is there at the end so that ks can be added to args without a special
  // case for time
  auto k0s =
      std::tuple_cat(std::make_tuple(call(func, args)),
                     squash_tuple<0, 2>(args), std::make_tuple(time_t(1)));

  args = init;
  // args to calculate k1s are obtained by adding k0s to initial values
  args = add_tuples(args, mult_tuple(k0s, step / time_t(2)));
  auto k1s =
      std::tuple_cat(std::make_tuple(call(func, args)),
                     squash_tuple<0, 2>(args), std::make_tuple(time_t(1)));

  args = init;
  // args to calculate k2s are obtained by adding k1s to initial values
  args = add_tuples(args, mult_tuple(k1s, step / time_t(2)));
  auto k2s =
      std::tuple_cat(std::make_tuple(call(func, args)),
                     squash_tuple<0, 2>(args), std::make_tuple(time_t(1)));

  args = init;
  // args to calculate k3s are obtained by adding k2s to initial values
  args = add_tuples(args, mult_tuple(k2s, step));
  auto k3s =
      std::tuple_cat(std::make_tuple(call(func, args)),
                     squash_tuple<0, 2>(args), std::make_tuple(time_t(1)));

  // calculate the new values from all ks using the formula y_(i+1) = y_i +
  // step/6 * (k0 + 2*k1 + 2*k2 + k3)
  auto final = add_tuples(k0s, mult_tuple(k1s, time_t(2)));
  final = add_tuples(final, mult_tuple(k2s, time_t(2)));
  final = add_tuples(final, std::move(k3s));
  final = mult_tuple(final, step / time_t(6));
  final = add_tuples(final, init);

  return final;
}

/* Given a function `func` expressing a derivative of order N, a time step size
 * `step`,
 * and N+1 values `vals` with the initial conditions of y(N-1), .., y1, y, t, in
 * that order,
 * returns the new values of y(N-1), ..., y1, y, t after the step in a tuple.
 * Uses Euler integration. */
template <typename time_t, typename func_t, typename... Args>
std::tuple<Args...> integrate_step_euler(func_t func, time_t step, Args... vals
                                         // TODO kmem ptr for temporary storage
) {
  static_assert(sizeof...(Args) >= 2,
                "Do you want infinite loops in your compiler? Because this is "
                "how you get infinite loops in your compiler.");

  auto const init = std::make_tuple(vals...);
  auto const to_add =
      std::tuple_cat(std::make_tuple(call(func, init)),
                     squash_tuple<0, 2>(init), std::make_tuple(time_t(1)));
  return add_tuples(init, mult_tuple(to_add, step));
}
