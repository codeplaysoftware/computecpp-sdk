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
 *  main.cpp
 *
 *  Description:
 *    Application description for Mandelbrot demo.
 *
 **************************************************************************/

#include <iostream>

#include <cinder/CameraUi.h>
#include <cinder/TriMesh.h>
#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <codeplay_demo.hpp>

#include "mandel.hpp"

constexpr size_t WIDTH = 800;
constexpr size_t HEIGHT = 600;

class MandelbrotApp
#ifdef CODEPLAY_DRAW_LOGO
    : public CodeplayDemoApp
#else
    : public ci::app::App
#endif
{
  // Use doubles for more zoom
  MandelbrotCalculator m_calc;

  // Texture for displaying the set
  ci::gl::Texture2dRef m_tex;

  // Coordinates of the center point
  double m_ctr_x = 0;
  double m_ctr_y = 0;

  // The viewable range on Y axis
  double m_range = 1;

  // Mouse coordinates from previous click
  double m_prev_mx = 0;
  double m_prev_my = 0;

 public:
  MandelbrotApp() : m_calc(WIDTH, HEIGHT) {}

  void setup() override {
    this->m_tex = ci::gl::Texture2d::create(nullptr, GL_RGBA, WIDTH, HEIGHT,
                                            ci::gl::Texture2d::Format()
                                                .dataType(GL_UNSIGNED_BYTE)
                                                .internalFormat(GL_RGBA));
  }

  void update() override {
    // Transform coordinates from the ones used here - center point
    // and range - to the ones used in MandelbrotCalculator - min and max X, Y.
    double range_x = m_range * double(WIDTH) / double(HEIGHT);
    auto half_x = range_x / 2.0f;
    double min_x = m_ctr_x - half_x;
    double max_x = m_ctr_x + half_x;
    auto half_y = m_range / 2.0f;
    double min_y = m_ctr_y - half_y;
    double max_y = m_ctr_y + half_y;

    // Set new coordinates and recalculate the fractal
    m_calc.set_bounds(min_x, max_x, min_y, max_y);
    if (m_calc.supports_doubles()) {
        m_calc.calc<double>();
    } else {
        m_calc.calc<float>();
    }
  }

  void draw() override {
    ci::gl::clear();

    // Update GL texture with new calculation data
    m_calc.with_data([&](sycl::cl_uchar4 const* data) {
      this->m_tex->update(data, GL_RGBA, GL_UNSIGNED_BYTE, 0, WIDTH, HEIGHT);
    });

    ci::Rectf screen(0, 0, getWindow()->getWidth(), getWindow()->getHeight());
    ci::gl::draw(m_tex, screen);

#ifdef CODEPLAY_DRAW_LOGO
    draw_codeplay_logo();
#endif
  }

  void mouseWheel(ci::app::MouseEvent event) override {
    // Zoom in or out on the plane
    auto inc = event.getWheelIncrement();
    if (inc > 0) {
      m_range *= 0.5f * inc;
    } else {
      m_range /= -0.5f * inc;
    }
  }

  void mouseDrag(ci::app::MouseEvent event) override {
    // Calculate normalized coordinates
    auto x = event.getX() / double(WIDTH);
    auto y = event.getY() / double(HEIGHT);

    // Find the difference from last click
    auto dx = m_prev_mx - x;
    // y coords are reversed
    auto dy = y - m_prev_my;

    // If the difference is big enough, drag the center point
    // and with it the viewable part of the plane. The epsilon
    // is necessary to avoid noisy jumps
    constexpr double EPS = .1;
    if (dx < EPS && dx > -EPS) {
      m_ctr_x += dx * m_range;
    }
    if (dy < EPS && dy > -EPS) {
      m_ctr_y += dy * m_range * double(WIDTH) / double(HEIGHT);
    }

    m_prev_mx = x;
    m_prev_my = y;
  }
};

CINDER_APP(MandelbrotApp, ci::app::RendererGl(ci::app::RendererGl::Options{}))
