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
 *  mandel.hpp
 *
 *  Description:
 *    SYCL kernel for Mandelbrot demo.
 *
 **************************************************************************/

#pragma once

#include <iostream>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;


/* Computes an image representing the Mandelbrot set on the complex
 * plane at a given zoom level. */
template <typename num_t>
class MandelbrotCalculator {
  // Dimensions of the image to be calculated
  size_t const m_width;
  size_t const m_height;

  // Accelerated SYCL queue and storage for image data
  sycl::queue m_q;
  sycl::buffer<sycl::cl_uchar4, 2> m_img;

  // Boundaries on the part of the complex plane which we want to view
  num_t m_minx = -2;
  num_t m_maxx = 1;
  num_t m_miny = -1;
  num_t m_maxy = 1;

 public:
  MandelbrotCalculator(size_t width, size_t height)
      : m_width(width),
        m_height(height),
        m_q(sycl::default_selector{},
            [&](sycl::exception_list el) {
              for (auto const& e : el) {
                try {
                  std::rethrow_exception(e);
                } catch (std::exception const& e) {
                  std::cout << "SYCL exception caught:\n:" << e.what()
                            << std::endl;
                }
              }
            }),
        // These are flipped since OpenGL expects column-major order for
        // textures
        m_img(sycl::range<2>(height, width)) {}

  // Set the boundaries of the viewable region. X is Re, Y is Im.
  void set_bounds(num_t min_x, num_t max_x, num_t min_y, num_t max_y) {
    m_minx = min_x;
    m_maxx = max_x;
    m_miny = min_y;
    m_maxy = max_y;
  }

  void calc();

  // Calls the function with the underlying image memory.
  template <typename Func>
  void with_data(Func&& func) {
    auto acc = m_img.get_access<sycl::access::mode::read>();

    func(acc.get_pointer());
  }

 private:
  void internal_calc() {
    m_q.submit([&](sycl::handler& cgh) {
      auto img_acc = m_img.get_access<sycl::access::mode::discard_write>(cgh);

      constexpr size_t MAX_ITERS = 500;
      // Anything above this number is assumed divergent. To do less
      // computation, this is the _square_ of the maximum absolute value
      // of a non-divergent number
      constexpr num_t DIVERGENCE_LIMIT = num_t(256);
      // Calculates how many iterations does it take to diverge? MAX_ITERS if in
      // Mandelbrot set
      const auto how_mandel = [](num_t re, num_t im) -> num_t {
        num_t z_re = 0;
        num_t z_im = 0;
        num_t abs_sq = 0;

        for (size_t i = 0; i < MAX_ITERS; i++) {
          num_t z_re2 = z_re * z_re - z_im * z_im + re;
          z_im = num_t(2) * z_re * z_im + im;
          z_re = z_re2;

          abs_sq = z_re * z_re + z_im * z_im;

          // Branching here isn't ideal, but it's the simplest
          if (abs_sq >= DIVERGENCE_LIMIT) {
            num_t log_zn = sycl::log(abs_sq) / num_t(2);
            num_t nu =
                sycl::log(log_zn / sycl::log(num_t(2))) / sycl::log(num_t(2));
            return num_t(i) + num_t(1) - nu;
          }
        }

        return num_t(1);
      };

      // Dummy variable copies to avoid capturing `this` in kernel lambda
      size_t width = m_width;
      size_t height = m_height;
      num_t minx = m_minx;
      num_t maxx = m_maxx;
      num_t miny = m_miny;
      num_t maxy = m_maxy;

      // Use the MandelbrotCalculator class for unique kernel name typ
      cgh.parallel_for<decltype(this)>(
          sycl::range<2>(m_height, m_width), [=](sycl::item<2> item) {
            // Obtain normalized coords [0, 1]
            num_t x = num_t(item.get_id(1)) / num_t(width);
            num_t y = num_t(item.get_id(0)) / num_t(height);

            // Put them within desired bounds
            x *= (maxx - minx);
            x += minx;

            y *= (maxy - miny);
            y += miny;

            // Calculate sequence divergence
            num_t mandelness = how_mandel(x, y);

            // Map to two colors in the palette
            const std::array<sycl::vec<num_t, 4>, 16> COLORS = {{
                {66, 30, 15, 255},
                {25, 7, 26, 255},
                {9, 1, 47, 255},
                {4, 4, 73, 255},
                {0, 7, 100, 255},
                {12, 44, 138, 255},
                {24, 82, 177, 255},
                {57, 125, 209, 255},
                {134, 181, 229, 255},
                {211, 236, 248, 255},
                {241, 233, 191, 255},
                {248, 201, 95, 255},
                {255, 170, 0, 255},
                {204, 128, 0, 255},
                {153, 87, 0, 255},
                {106, 52, 3, 255},
            }};

            auto col_a = COLORS[size_t(mandelness) % COLORS.size()];
            auto col_b = COLORS[(size_t(mandelness) + 1) % COLORS.size()];

            // fract(a) = a - floor(a)
            auto fract = mandelness - num_t(size_t(mandelness));

            // Linearly interpolate between the colors using the fractional part
            // of 'mandelness' to get smooth transitions
            auto col =
                col_a * (num_t(1) - fract) +
                col_b * fract;

            // Store color in image
            img_acc[item] = { col.x(), col.y(), col.z(), col.w() };
          });
    });
  }
};
