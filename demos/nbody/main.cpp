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

#include <chrono>
#include <iostream>

#include <cinder/CameraUi.h>
#include <cinder/TriMesh.h>
#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>

#include <cinder/CinderImGui.h>

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
    float min_height = -50;
    float max_height = 50;
    float lg_speed = 0.4;
  } m_ui_distrib_cylinder_params;

  struct {
    float min_radius = 0;
    float max_radius = 25;
  } m_ui_distrib_sphere_params;

  int32_t m_ui_n_bodies = 1024;

  const int32_t m_num_updates_per_frame = 1;
  int32_t m_num_updates = 0;

  // Whether 'initialize' was clicked
  bool m_ui_initialize = false;

  // Whether the simulation is paused
  bool m_ui_paused = true;

  // Whether the user has requested a single step to be computed
  bool m_ui_step = false;

  // Force choice
  enum {
    UI_FORCE_GRAVITY = 0,
    UI_FORCE_LJ = 1,
    UI_FORCE_COULOMB = 2,
  };
  int32_t m_ui_force_id = UI_FORCE_GRAVITY;

  // Force parameters
  struct {
    float lg_G = -1.4;
    float lg_damping = -3;
  } m_ui_force_gravity_params;

  struct {
    float eps = 1;
    float lg_sigma = -5;
  } m_ui_force_lj_params;

  std::array<char, 256> m_ui_force_coulomb_file;

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
                in vec3 ciColor; out vec3 vColor;

                vec3 speedColors[9] =
                    vec3[](vec3(0.0, 0.0, 0.2), vec3(0.0, 0.0, 0.4),
                           vec3(0.0, 0.0, 0.8), vec3(0.0, 0.4, 0.4),
                           vec3(0.0, 0.8, 0.8), vec3(0.0, 0.8, 0.4),
                           vec3(0.4, 0.8, 0.0), vec3(0.8, 0.6, 0.0),
                           vec3(0.8, 0.2, 0.0));

                void main() {
                  vec4 viewSpacePos = ciModelView * vec4(ciPosition, 1.0);
                  // The further away a point is, the smaller its sprite
                  float scale = log2(length(viewSpacePos));
                  gl_PointSize = 10.0 / clamp(scale, 0.1, 1);
                  gl_Position =
                      ciModelViewProjection * vec4(ciPosition / 10.0, 1.0);
                  float len = length(ciColor);
                  int speed = int(ceil(len));
                  vColor = mix(speedColors[max(0, speed)],
                               speedColors[min(8, speed)], speed - len);
                }))
            .fragment(CI_GLSL(
                330, uniform sampler2D star_tex; in vec3 vColor;
                out vec4 oColor; void main() {
                  vec2 uv = vec2(gl_PointCoord.x, gl_PointCoord.y);
                  vec4 tex = texture2D(star_tex, uv);
                  oColor = vec4(vColor, tex.a);
                })));

    // Make the texture uniform use slot 0, which we bound the texture to
    m_shader->uniform("star_tex", 0);

    m_ui_force_coulomb_file.fill(0);
    ImGui::Initialize();

    // Start with a perspective camera looking at the center
    m_cam.lookAt(ci::vec3(200), ci::vec3(0));
    m_cam.setPerspective(60.0f, getWindowAspectRatio(), 0.01f, 50000.0f);
    m_cam_ui = ci::CameraUi(&m_cam, getWindow());

    init_gl_bufs();

    // Start the simulation with the application
    m_ui_initialize = true;
    m_ui_paused = false;
  }

  // Initializes the GL buffer for star position data with the current number of
  // bodies
  void init_gl_bufs() {
    auto layout = ci::geom::BufferLayout();
    layout.append(ci::geom::Attrib::POSITION, 3, sizeof(sycl::vec<num_t, 3>),
                  0);
    layout.append(ci::geom::Attrib::COLOR, 3, sizeof(sycl::vec<num_t, 3>),
                  m_n_bodies * sizeof(sycl::vec<num_t, 3>));

    m_vbo = cinder::gl::Vbo::create(GL_ARRAY_BUFFER);
    m_vbo->bufferData(m_n_bodies * sizeof(sycl::vec<num_t, 3>) * 2, nullptr,
                      GL_DYNAMIC_DRAW);

    auto mesh = cinder::gl::VboMesh::create(m_n_bodies, GL_POINTS,
                                            {std::make_pair(layout, m_vbo)});
    m_batch = ci::gl::Batch::create(mesh, m_shader);
  }

  void update() override {
    // Initialize simulation if requested in UI
    if (m_ui_initialize) {
      m_n_bodies = m_ui_n_bodies;

      if (m_ui_force_id == UI_FORCE_COULOMB) {
        printf("Loading Coulomb data from %s\n",
               m_ui_force_coulomb_file.data());

        // First line is the particle count
        std::ifstream fin{m_ui_force_coulomb_file.data()};
        fin >> m_n_bodies;

        // Following lines are particle data
        std::vector<particle_data<num_t>> particles(m_n_bodies);
        for (auto& particle : particles) {
          num_t pos[3];
          fin >> particle.charge >> pos[0] >> pos[1] >> pos[2];
          particle.pos = {pos[0], pos[1], pos[2]};
        }

        m_sim = GravSim<num_t>(m_n_bodies, std::move(particles));
      } else if (m_ui_distrib_id == UI_DISTRIB_CYLINDER) {
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

    if (!m_ui_paused || m_ui_step) {
      // Update force parameters
      switch (m_ui_force_id) {
        case UI_FORCE_GRAVITY: {
          m_sim.set_grav_G(
              sycl::pow(num_t(10), m_ui_force_gravity_params.lg_G));
          m_sim.set_grav_damping(
              sycl::pow(num_t(10), m_ui_force_gravity_params.lg_damping));
          m_sim.set_force_type(force_t::GRAVITY);
        } break;

        case UI_FORCE_LJ: {
          m_sim.set_lj_eps(m_ui_force_lj_params.eps);
          m_sim.set_lj_sigma(
              sycl::pow(num_t(10), m_ui_force_lj_params.lg_sigma));
          m_sim.set_force_type(force_t::LENNARD_JONES);
        } break;

        case UI_FORCE_COULOMB: {
          m_sim.set_force_type(force_t::COULOMB);
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
      if (m_ui_step) {
        m_sim.sync_queue();

        // Measure submission, execution and sync
        auto tstart = std::chrono::high_resolution_clock::now();
        m_sim.step();
        m_sim.sync_queue();
        auto tend = std::chrono::high_resolution_clock::now();

        // Convert to seconds
        auto diff = tend - tstart;
        auto sdiff = std::chrono::duration_cast<
                         std::chrono::duration<num_t, std::ratio<1, 1>>>(diff)
                         .count();

        std::cout << "Time taken for step: " << sdiff << "s" << std::endl;
      } else {
        for (int32_t step = 0; step < m_num_updates_per_frame; ++step) {
          m_sim.step();
        }
      }

      // Make sure not to step until clicked again
      m_ui_step = false;
    }

    m_num_updates++;
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
    m_sim.with_mapped(
        read_bufs_t<0>{}, [&](sycl::vec<num_t, 3> const* velocities) {
          m_vbo->bufferSubData(m_n_bodies * sizeof(sycl::vec<num_t, 3>),
                               m_n_bodies * sizeof(sycl::vec<num_t, 3>),
                               velocities);
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
    ImGui::Begin("Simulation Settings");

    std::array<const char*, 3> forces = {
        {"Gravity", "Lennard-Jones", "Coulomb"}};
    ImGui::ListBox("Type of force", &m_ui_force_id, forces.data(),
                   forces.size(), forces.size());

    std::array<const char*, 2> integrators = {
        {"Euler [fast, inaccurate]", "RK4 [slow, accurate]"}};
    ImGui::ListBox("Integrator", &m_ui_integrator_id, integrators.data(),
                   integrators.size(), integrators.size());

    switch (m_ui_force_id) {
      case UI_FORCE_GRAVITY: {
        if (ImGui::TreeNode("Gravity settings")) {
          ImGui::SliderFloat("G constant [lg]", &m_ui_force_gravity_params.lg_G,
                             -8, 2);

          ImGui::SliderFloat("Damping factor [lg]",
                             &m_ui_force_gravity_params.lg_damping, -14, 0);

          ImGui::TreePop();
        }
      } break;

      case UI_FORCE_LJ: {
        if (ImGui::TreeNode("Lennard-Jones settings")) {
          ImGui::SliderFloat("Potential well depth", &m_ui_force_lj_params.eps,
                             0.1, 10);

          ImGui::SliderFloat("Zero potential radius [lg]",
                             &m_ui_force_lj_params.lg_sigma, -8, -2);

          ImGui::TreePop();
        }
      } break;

      case UI_FORCE_COULOMB: {
        if (ImGui::TreeNode("Coulomb settings")) {
          ImGui::Text(
              "Data format:\nLine 1: particle count (N)\nLines 2-(N+1): "
              "<charge> <x> <y> <z>");
          ImGui::InputText("Data input file", m_ui_force_coulomb_file.data(),
                           m_ui_force_coulomb_file.size());

          if (ImGui::Button("Initialize from file")) {
            m_ui_initialize = true;
          }

          ImGui::TreePop();
        }

      } break;

      default:
        throw std::runtime_error("unreachable");
    }

    if (ImGui::TreeNode("Initialization")) {
      std::array<const char*, 2> distribs = {{"Cylinder", "Sphere"}};
      ImGui::ListBox("Distribution", &m_ui_distrib_id, distribs.data(),
                     distribs.size(), distribs.size());

      ImGui::SliderInt("Number of bodies", &m_ui_n_bodies, 128, 16384);

      switch (m_ui_distrib_id) {
        case UI_DISTRIB_CYLINDER: {
          if (ImGui::TreeNode("Cylinder distribution settings")) {
            ImGui::DragFloatRange2(
                "Radius", &m_ui_distrib_cylinder_params.min_radius,
                &m_ui_distrib_cylinder_params.max_radius, 0.1f, 0.0f, 100.0f);

            ImGui::DragFloatRange2(
                "Angle [pi]", &m_ui_distrib_cylinder_params.min_angle_pis,
                &m_ui_distrib_cylinder_params.max_angle_pis, 0.01f, 0.0f, 2.0f);

            ImGui::DragFloatRange2("Height",
                                   &m_ui_distrib_cylinder_params.min_height,
                                   &m_ui_distrib_cylinder_params.max_height,
                                   0.1f, -100.0f, 100.0f);

            ImGui::SliderFloat("Speed [lg]",
                               &m_ui_distrib_cylinder_params.lg_speed, -3, 1);

            if (ImGui::Button("Initialize from distribution")) {
              m_ui_initialize = true;
            }

            ImGui::TreePop();
          }
        } break;

        case UI_DISTRIB_SPHERE: {
          if (ImGui::TreeNode("Sphere distribution settings")) {
            ImGui::DragFloatRange2(
                "Radius", &m_ui_distrib_sphere_params.min_radius,
                &m_ui_distrib_sphere_params.max_radius, 0.1f, 0.0f, 100.0f);

            if (ImGui::Button("Initialize from distribution")) {
              m_ui_initialize = true;
            }

            ImGui::TreePop();
          }
        } break;

        default:
          throw std::runtime_error("unreachable");
      }

      ImGui::TreePop();
    }

    if (m_ui_paused) {
      if (ImGui::Button("Start")) {
        m_ui_paused = false;
      }
      if (ImGui::Button("Step")) {
        m_ui_step = true;
      }
    } else {
      if (ImGui::Button("Pause")) {
        m_ui_paused = true;
      }
    }

    ImGui::End();
  }
};

CINDER_APP(NBodyApp, ci::app::RendererGl(ci::app::RendererGl::Options{}))
