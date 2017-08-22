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
 *    Application description for NBody demo.
 *
 **************************************************************************/

#include <iostream>

#include <cinder/CameraUi.h>
#include <cinder/TriMesh.h>
#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>

#include <CinderImGui.h>

#include <CL/sycl.hpp>

#include <codeplay_demo.hpp>

#include "sim.hpp"


using num_t = float;
constexpr num_t PI = num_t(3.141592653589793238462643383279502884197169399);

class NBodyApp
#ifdef CODEPLAY_DRAW_LOGO
    : public CodeplayDemoApp
#else
    : public ci::app::App
#endif
{
  // -- GUI --
  // Distribution choice
  enum {
    UI_DISTRIB_CYLINDER = 0,
    UI_DISTRIB_SPHERE = 1,
  };
  int32_t m_ui_distrib_id = UI_DISTRIB_CYLINDER;

  // Distribution parameters
  struct {
    float min_radius = 0;
    float max_radius = 25;
    float min_angle_pis = 0;
    float max_angle_pis = 2;
    float min_height = -5;
    float max_height = 5;
    float lg_speed = 0;
  } m_ui_distrib_cylinder_params;

  struct {
    float min_radius = 0;
    float max_radius = 25;
  } m_ui_distrib_sphere_params;

  int32_t m_ui_n_bodies = 128;

  // Whether 'initialize' was clicked
  bool m_ui_initialize = false;

  // Force choice
  enum {
    UI_FORCE_GRAVITY = 0,
    UI_FORCE_LJ = 1,
  };
  int32_t m_ui_force_id = UI_FORCE_GRAVITY;

  // Force parameters
  struct {
    float lg_G = -3;
    float lg_damping = -3;
  } m_ui_force_gravity_params;

  struct {
    float eps = 1;
    float lg_sigma = -5;
  } m_ui_force_lj_params;

  // Integrator choice
  enum {
    UI_INTEGRATOR_EULER = 0,
    UI_INTEGRATOR_RK4 = 1,
  };
  int32_t m_ui_integrator_id = UI_INTEGRATOR_EULER;

  // -- PROGRAM VARIABLES --
  size_t m_n_bodies = m_ui_n_bodies;

  // Cinder-GL variables
  ci::CameraPersp m_cam;
  ci::CameraUi m_cam_ui;
  ci::gl::VboRef m_vbo;
  ci::gl::Texture2dRef m_star_tex;
  ci::gl::GlslProgRef m_shader;
  ci::gl::BatchRef m_batch;

  // The simulation
  GravSim<num_t> m_sim;

 public:
  NBodyApp() : m_sim(m_n_bodies, distrib_cylinder<num_t>{}) {}

  void setup() override {
    // Create star texture and bind it to GL slot 0
    m_star_tex = ci::gl::Texture2d::create(loadImage(loadAsset("star.png")));
    m_star_tex->bind(0);

    // Create shader to display the stars
    m_shader = ci::gl::GlslProg::create(
        ci::gl::GlslProg::Format()
            .vertex(CI_GLSL(
                330, uniform mat4 ciModelView;
                uniform mat4 ciModelViewProjection; in vec3 ciPosition;

                void main() {
                  vec4 viewSpacePos = ciModelView * vec4(ciPosition, 1.0);
                  // The further away a point is, the smaller its sprite
                  float scale = log2(length(viewSpacePos));
                  gl_PointSize = 10.0 / clamp(scale, 0.1, 1);
                  gl_Position = ciModelViewProjection * vec4(ciPosition, 1.0);
                }))
            .fragment(CI_GLSL(
                330, uniform sampler2D star_tex; out vec4 oColor; void main() {
                  vec2 uv = vec2(gl_PointCoord.x, gl_PointCoord.y);
                  vec4 tex = texture2D(star_tex, uv);
                  oColor = vec4(vec3(212.0, 175.0, 55.0) / 255.0, tex.a);
                })));

    // Make the texture uniform use slot 0, which we bound the texture to
    m_shader->uniform("star_tex", 0);

    // Start with a perspective camera looking at the center
    m_cam.lookAt(ci::vec3(200), ci::vec3(0));
    m_cam.setPerspective(60.0f, getWindowAspectRatio(), 0.01f, 5000.0f);

    ui::initialize();
    m_cam_ui = ci::CameraUi(&m_cam, getWindow());

    init_gl_bufs();
  }

  // Initializes the GL buffer for star position data with the current number of
  // bodies
  void init_gl_bufs() {
    auto layout = ci::geom::BufferLayout();
    layout.append(ci::geom::Attrib::POSITION, 3, sizeof(sycl::vec<num_t, 3>),
                  0);

    m_vbo = cinder::gl::Vbo::create(GL_ARRAY_BUFFER);
    m_vbo->bufferData(m_n_bodies * sizeof(sycl::vec<num_t, 3>), nullptr,
                      GL_DYNAMIC_DRAW);

    auto mesh = cinder::gl::VboMesh::create(m_n_bodies, GL_POINTS,
                                            {std::make_pair(layout, m_vbo)});
    m_batch = ci::gl::Batch::create(mesh, m_shader);
  }

  void update() override {
    // Initialize simulation if requested in UI
    if (m_ui_initialize) {
      m_n_bodies = m_ui_n_bodies;

      if (m_ui_distrib_id == UI_DISTRIB_CYLINDER) {
        m_sim = GravSim<num_t>(
            m_n_bodies,
            distrib_cylinder<num_t>{
                {m_ui_distrib_cylinder_params.min_radius,
                 m_ui_distrib_cylinder_params.max_radius},
                {m_ui_distrib_cylinder_params.min_angle_pis * PI,
                 m_ui_distrib_cylinder_params.max_angle_pis * PI},
                {m_ui_distrib_cylinder_params.min_height,
                 m_ui_distrib_cylinder_params.max_height},
                sycl::pow(num_t(10), m_ui_distrib_cylinder_params.lg_speed)});
      } else if (m_ui_distrib_id == UI_DISTRIB_SPHERE) {
        m_sim = GravSim<num_t>(
            m_n_bodies,
            distrib_sphere<num_t>{{m_ui_distrib_sphere_params.min_radius,
                                   m_ui_distrib_sphere_params.max_radius}});
      }

      init_gl_bufs();

      m_ui_initialize = false;
    }

    // Update force parameters
    switch (m_ui_force_id) {
      case UI_FORCE_GRAVITY: {
        m_sim.set_grav_G(sycl::pow(num_t(10), m_ui_force_gravity_params.lg_G));
        m_sim.set_grav_damping(
            sycl::pow(num_t(10), m_ui_force_gravity_params.lg_damping));
        m_sim.set_force_type(force_t::GRAVITY);
      } break;

      case UI_FORCE_LJ: {
        m_sim.set_lj_eps(m_ui_force_lj_params.eps);
        m_sim.set_lj_sigma(sycl::pow(num_t(10), m_ui_force_lj_params.lg_sigma));
        m_sim.set_force_type(force_t::LENNARD_JONES);
      } break;

      default:
        throw "unreachable";
    }

    // Update integration method
    switch (m_ui_integrator_id) {
      case UI_INTEGRATOR_EULER: {
        m_sim.set_integrator(integrator_t::EULER);
      } break;

      case UI_INTEGRATOR_RK4: {
        m_sim.set_integrator(integrator_t::RK4);
      } break;

      default:
        throw "unreachable";
    }

    // Run simulation frame
    m_sim.step();
  }

  void draw() override {
    ci::gl::clear();

    // Disable depth to avoid black outlines
    ci::gl::disableDepthRead();
    ci::gl::disableDepthWrite();

    // Colors add to a bright white with additive blending
    ci::gl::enableAdditiveBlending();

    // Set transform to the camera view
    ci::gl::setMatrices(m_cam);

    // Update star buffer data with new positions
    m_sim.with_mapped(
        read_bufs_t<1>{}, [&](sycl::vec<num_t, 3> const* positions) {
          m_vbo->bufferSubData(0, m_n_bodies * sizeof(sycl::vec<num_t, 3>),
                               positions);
        });

    // Enable setting point size in shader
    ci::gl::enable(GL_VERTEX_PROGRAM_POINT_SIZE, true);

    // Draw bodies
    m_batch->draw();

    // Draw coordinate system arrows
    ci::gl::color(ci::Color(1.0f, 0.0f, 0.0f));
    ci::gl::drawVector(ci::vec3(90, 0, 0), ci::vec3(100, 0, 0), 2, 2);
    ci::gl::color(ci::Color(0.0f, 1.0f, 0.0f));
    ci::gl::drawVector(ci::vec3(0, 90, 0), ci::vec3(0, 100, 0), 2, 2);
    ci::gl::color(ci::Color(0.0f, 0.0f, 1.0f));
    ci::gl::drawVector(ci::vec3(0, 0, 90), ci::vec3(0, 0, 100), 2, 2);

#ifdef CODEPLAY_DRAW_LOGO
    draw_codeplay_logo();
#endif

    // Draw the UI
    ui::Begin("Simulation Settings");

    std::array<const char*, 2> forces = {{"Gravity", "Lennard-Jones"}};
    ui::ListBox("Type of force", &m_ui_force_id, forces.data(), forces.size(),
                forces.size());

    std::array<const char*, 2> integrators = {
        {"Euler [fast, inaccurate]", "RK4 [slow, accurate]"}};
    ui::ListBox("Integrator", &m_ui_integrator_id, integrators.data(),
                integrators.size(), integrators.size());

    switch (m_ui_force_id) {
      case UI_FORCE_GRAVITY: {
        if (ui::TreeNode("Gravity settings")) {
          ui::SliderFloat("G constant [lg]", &m_ui_force_gravity_params.lg_G,
                          -8, 2);

          ui::SliderFloat("Damping factor [lg]",
                          &m_ui_force_gravity_params.lg_damping, -14, 0);

          ui::TreePop();
        }
      } break;

      case UI_FORCE_LJ: {
        if (ui::TreeNode("Lennard-Jones settings")) {
          ui::SliderFloat("Potential well depth", &m_ui_force_lj_params.eps,
                          0.1, 10);

          ui::SliderFloat("Zero potential radius [lg]",
                          &m_ui_force_lj_params.lg_sigma, -8, -2);

          ui::TreePop();
        }
      } break;

      default:
        throw std::runtime_error("unreachable");
    }

    if (ui::TreeNode("Initial")) {
      std::array<const char*, 2> distribs = {{"Cylinder", "Sphere"}};
      ui::ListBox("Distribution", &m_ui_distrib_id, distribs.data(),
                  distribs.size(), distribs.size());

      ui::SliderInt("Number of bodies", &m_ui_n_bodies, 128, 16384);

      switch (m_ui_distrib_id) {
        case UI_DISTRIB_CYLINDER: {
          if (ui::TreeNode("Cylinder distribution settings")) {
            ui::DragFloatRange2(
                "Radius", &m_ui_distrib_cylinder_params.min_radius,
                &m_ui_distrib_cylinder_params.max_radius, 0.1f, 0.0f, 100.0f);

            ui::DragFloatRange2(
                "Angle [pi]", &m_ui_distrib_cylinder_params.min_angle_pis,
                &m_ui_distrib_cylinder_params.max_angle_pis, 0.01f, 0.0f, 2.0f);

            ui::DragFloatRange2("Height",
                                &m_ui_distrib_cylinder_params.min_height,
                                &m_ui_distrib_cylinder_params.max_height, 0.1f,
                                -100.0f, 100.0f);

            ui::SliderFloat("Speed [lg]",
                            &m_ui_distrib_cylinder_params.lg_speed, -3, 1);

            ui::TreePop();
          }
        } break;

        case UI_DISTRIB_SPHERE: {
          if (ui::TreeNode("Sphere distribution settings")) {
            ui::DragFloatRange2(
                "Radius", &m_ui_distrib_sphere_params.min_radius,
                &m_ui_distrib_sphere_params.max_radius, 0.1f, 0.0f, 100.0f);

            ui::TreePop();
          }
        } break;

        default:
          throw std::runtime_error("unreachable");
      }

      if (ui::Button("Initialize")) {
        m_ui_initialize = true;
      }

      ui::TreePop();
    }

    ui::End();
  }
};

CINDER_APP(NBodyApp, ci::app::RendererGl(ci::app::RendererGl::Options{}))
