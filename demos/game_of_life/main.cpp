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
 *    Application description for Game of Life demo.
 *
 **************************************************************************/

#include <cinder/CameraUi.h>
#include <cinder/TriMesh.h>
#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <codeplay_demo.hpp>

#include "sim.hpp"

class GameOfLifeApp
#ifdef CODEPLAY_DRAW_LOGO
    : public CodeplayDemoApp
#else
    : public ci::app::App
#endif
{
  /// The window dimensions
  size_t m_width;
  size_t m_height;
  float m_zoom;

  /// Things to do with resizing the window. Have to be done to make it smooth.
  /// Wait this many frames before reinitializing after resize
  size_t RESIZE_TIMEOUT = 15;
  /// Counter for reinitializing after resize
  size_t m_resize_time = 0;
  /// Whether a resize has been executed
  size_t m_resized = false;
  /// Dimensions after the resize
  size_t m_resized_width;
  size_t m_resized_height;

  /// Whether the simulation is running or paused
  bool m_paused = false;

  /// The simulation
  GameOfLifeSim m_sim;

  /// The texture to display the simulation on
  ci::gl::Texture2dRef m_tex;

 public:
  GameOfLifeApp()
      : m_width(getWindow()->getWidth()),
        m_height(getWindow()->getHeight()),
        m_zoom(1),
        m_sim(m_width, m_height) {}

  void setup() override {
    // Initializes image data
    m_tex = ci::gl::Texture2d::create(nullptr, GL_RGBA, m_width, m_height,
                                      ci::gl::Texture2d::Format()
                                          .dataType(GL_UNSIGNED_BYTE)
                                          .internalFormat(GL_RGBA));
  }

  void update() override {
    if (m_resized) {
      if (m_resize_time++ >= RESIZE_TIMEOUT) {
        m_width = m_resized_width / m_zoom;
        m_height = m_resized_height / m_zoom;

        m_sim = GameOfLifeSim(m_width, m_height);
        // Reinitializes image to new size
        m_tex = ci::gl::Texture2d::create(nullptr, GL_RGBA, m_width, m_height,
                                          ci::gl::Texture2d::Format()
                                              .dataType(GL_UNSIGNED_BYTE)
                                              .internalFormat(GL_RGBA));
        m_tex->setMagFilter(GL_NEAREST);

        m_resized = false;
        m_resize_time = 0;
      }
    }

    if (!m_paused) {
      m_sim.step();
      // 60 FPS
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
  }

  void draw() override {
    ci::gl::clear();

    m_sim.with_img(
        // Gets called with image data
        [&](sycl::cl_uchar4 const* data) {
          this->m_tex->update(data, GL_RGBA, GL_UNSIGNED_BYTE, 0, m_width,
                              m_height);
        });

    ci::Rectf screen(0, 0, getWindow()->getWidth(), getWindow()->getHeight());
    ci::gl::draw(this->m_tex, screen);

#ifdef CODEPLAY_DRAW_LOGO
    draw_codeplay_logo();
#endif
  }

  void handleMouse(size_t mouse_x, size_t mouse_y) {
    // Obtain coordinates within the simulation dimensions
    size_t x = static_cast<float>(mouse_x) /
               static_cast<float>(getWindow()->getWidth()) *
               static_cast<float>(m_width);
    size_t y = static_cast<float>(mouse_y) /
               static_cast<float>(getWindow()->getHeight()) *
               static_cast<float>(m_height);
    // Invert Y
    y = m_height - y;
    if (x < m_width && y < m_height) {
      // Set cell at mouse position to alive
      m_sim.add_click(x, y + 1, CellState::LIVE);
      m_sim.add_click(x + 1, y, CellState::LIVE);
      m_sim.add_click(x, y - 1, CellState::LIVE);
      m_sim.add_click(x - 1, y - 1, CellState::LIVE);      
    }    
  }

  void mouseDown(ci::app::MouseEvent event) override {
    handleMouse(event.getX(), event.getY());
  }

  void mouseDrag(ci::app::MouseEvent event) override {
    handleMouse(event.getX(), event.getY());
  }

  void mouseWheel(ci::app::MouseEvent event) override {
    auto inc = event.getWheelIncrement();
    if (inc > 0) {
      // Zoom in on wheel up
      m_zoom *= 2.0f * inc;
    } else {
      // Zoom out on wheel down
      m_zoom /= -2.0f * inc;
    }
    // Don't zoom out further than one cell per pixel
    m_zoom = sycl::clamp(m_zoom, 1.0f, 64.0f);

    // Restart after zooming
    m_resized = true;
  }

  void keyDown(ci::app::KeyEvent event) override {
    // (Un)pause on SPACE
    if (event.getCode() == ci::app::KeyEvent::KEY_SPACE) {
      m_paused = !m_paused;
    }
  }

  void resize() override {
    auto const& win = getWindow();
    m_resized_width = win->getWidth();
    m_resized_height = win->getHeight();
    m_resized = true;
  }
};

CINDER_APP(GameOfLifeApp, ci::app::RendererGl(ci::app::RendererGl::Options{}))
