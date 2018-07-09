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
 *  sim.hpp
 *
 *  Description:
 *    SYCL kernel for NBody demo.
 *
 **************************************************************************/

#pragma once

#include <iostream>
#include <random>
#include <memory>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <integrator.hpp>

#include <double_buf.hpp>
#include <sycl_bufs.hpp>
#include <tuple_utils.hpp>


// Convenience types
template <typename num_t>
using vec3 = sycl::vec<num_t, 3>;

// Template to generate unique kernel name types
template <typename T, size_t Z>
class kernel {};

// Handler for queue exceptions
const auto except_handler = [](sycl::exception_list el) {
	                            for (std::exception_ptr const& e : el) {
		                            try {
			                            std::rethrow_exception(e);
		                            } catch (sycl::exception& e) {
			                            std::cout << "EXCEPTION:\n" << e.what() << std::endl;
		                            }
	                            }
                            };

// Initial cylinder distribution parameters
template <typename num_t>
struct distrib_cylinder {
	sycl::vec<num_t, 2> radius;
	sycl::vec<num_t, 2> angle;
	sycl::vec<num_t, 2> height;
	num_t speed;
};

// Data describing a single charged particle
template <typename num_t>
struct particle_data {
	num_t charge;
	sycl::vec<num_t, 3> pos;
};

// Initial sphere distribution parameters
template <typename num_t>
struct distrib_sphere {
	sycl::vec<num_t, 2> radius;
};

// The kind of force to simulate
enum class force_t {
	GRAVITY,
	LENNARD_JONES,
	COULOMB,
};

// Which integration method to use
enum class integrator_t {
	EULER,
	RK4,
};

template <typename num_t>
class GravSim {
	sycl::queue m_q;

	// Buffers storing body data: (velocity, position)
	DoubleBuf<SyclBufs<vec3<num_t>, vec3<num_t>>> m_bufs;

	// Buffer for charges used only in Coulomb simulation
	std::unique_ptr<SyclBufs<num_t>> m_coulomb_charges_buf = nullptr;

	// The number of bodies partaking in the simulation
	size_t m_n_bodies;

	// The size of a single timestep
	static constexpr num_t STEP_SIZE = num_t(.5);

	// The current time of the simulation
	num_t m_time;

	// Which force we are simulating
	force_t m_force;

	// Force parameters
	struct {
		num_t G = 1e-5;
		num_t damping = 1e-5;
	} m_grav_params;

	struct {
		num_t eps = 1;
		num_t sigma = 1e-3;
	} m_lj_params;

	// Which integrator to use
	integrator_t m_integrator;

	// Base constructor, does not initialize simulation values
	GravSim(size_t n_bodies)
		: m_q(sycl::default_selector{}, except_handler),
		  m_bufs(n_bodies),
		  m_n_bodies(n_bodies),
		  m_time(0),
		  m_force(force_t::GRAVITY) {}

public:
	// Initialize the simulation with a cylinder body distribution
	GravSim(size_t n_bodies, distrib_cylinder<num_t> params) : GravSim(n_bodies) {
		// Generates points uniformly distributed in a cylinder using cylindrical
		// polar coordinates
		std::mt19937 rng(std::random_device{}());
		auto rmin = params.radius.x();
		auto rmax = params.radius.y();
		std::uniform_real_distribution<num_t> unifr(rmin * rmin, rmax * rmax);
		std::uniform_real_distribution<num_t> unifp(params.angle.x(),
		                                            params.angle.y());
		std::uniform_real_distribution<num_t> unify(params.height.x(),
		                                            params.height.y());

		auto accs = m_bufs.write().gen_host_write_accs(write_bufs_t<0, 1>{});
		for (size_t i = 0; i < m_n_bodies; i++) {
			auto r = sycl::sqrt(unifr(rng));
			auto phi = unifp(rng);

			// Velocity tangential to the circular cylinder slice is given
			// by the derivative of position w.r.t phi
			std::get<0>(accs)[i] = {-r * sycl::sin(phi), num_t(0),
				r * sycl::cos(phi)};
			// Adjust to make chosen speed the speed of outermost bodies
			std::get<0>(accs)[i] *= params.speed / params.radius.y();
			std::get<1>(accs)[i] = {r * sycl::cos(phi), unify(rng),
				r * sycl::sin(phi)};
		}

		// Make newly-written data the read-buffer
		m_bufs.swap();
	}

	// Initialize the simulation with a sphere body distribution
	GravSim(size_t n_bodies, distrib_sphere<num_t> params) : GravSim(n_bodies) {
		// Generates a uniform spherical distribution from spherical coordinates
		std::mt19937 rng(std::random_device{}());
		std::uniform_real_distribution<num_t> unifp(0, 2 * 3.141592f);
		std::uniform_real_distribution<num_t> unifcost(-1, 1);
		auto rmin = params.radius.x();
		auto rmax = params.radius.y();
		std::uniform_real_distribution<num_t> unifu(rmin * rmin * rmin,
		                                            rmax * rmax * rmax);

		auto accs = m_bufs.write().gen_host_write_accs(write_bufs_t<0, 1>{});

		for (size_t i = 0; i < m_n_bodies; i++) {
			auto r = sycl::pow(unifu(rng), num_t(1) / num_t(3));
			auto cost = unifcost(rng);
			auto sint = sycl::sqrt(1 - cost * cost);
			auto phi = unifp(rng);
			auto x = r * sint * sycl::cos(phi);
			auto y = r * sint * sycl::sin(phi);
			auto z = r * cost;

			// Spherical distribution gives no initial velocity to bodies
			std::get<0>(accs)[i] = {0, 0, 0};
			std::get<1>(accs)[i] = {x, y, z};
		}

		// Make newly-written data the read-buffer
		m_bufs.swap();
	}

	GravSim(size_t n_bodies, std::vector<particle_data<num_t>>&& particles) : GravSim(n_bodies) {
		m_coulomb_charges_buf = std::unique_ptr<SyclBufs<num_t>>(new SyclBufs<num_t>(n_bodies));

		auto accs = m_bufs.write().gen_host_write_accs(write_bufs_t<0, 1>{});
		auto charge_acc = m_coulomb_charges_buf->gen_host_write_accs(write_bufs_t<0>{});

		for (size_t i = 0; i < m_n_bodies; i++) {
			std::get<0>(charge_acc)[i] = particles[i].charge;

			std::get<0>(accs)[i] = {0, 0, 0};
			std::get<1>(accs)[i] = particles[i].pos;
		}

		// Make newly-written data the read-buffer
		m_bufs.swap();
	}

	void step();

	void sync_queue() { m_q.wait(); }

	void set_force_type(force_t force) { m_force = force; }

	void set_integrator(integrator_t integrator) { m_integrator = integrator; }

	// Set gravity damping
	void set_grav_damping(num_t damping) { m_grav_params.damping = damping; }

	// Set gravitational constant
	void set_grav_G(num_t G) { m_grav_params.G = G; }

	// Set Lennard-Jones potential well depth
	void set_lj_eps(num_t eps) { m_lj_params.eps = eps; }

	// Set Lennard-Jones zero-potential distance
	void set_lj_sigma(num_t sigma) { m_lj_params.sigma = sigma; }

	// Calls the provided function with body position data
	template <typename Func, size_t VarId>
	void with_mapped(read_bufs_t<VarId>, Func&& func) {
		auto acc = m_bufs.read().gen_host_read_accs(read_bufs_t<VarId>{});
		func(std::get<0>(acc).get_pointer());
	}

private:
	void internal_step() {
		m_q.submit([&](cl::sycl::handler& cgh) {
			           // Initialize accessors to body data
			           auto reads = m_bufs.read().gen_read_accs(cgh, read_bufs_t<0, 1>{});
			           auto writes = m_bufs.write().gen_write_accs(cgh, write_bufs_t<0, 1>{});
			           auto vel = std::get<0>(reads);
			           auto pos = std::get<1>(reads);
			           auto wvel = std::get<0>(writes);
			           auto wpos = std::get<1>(writes);

			           // Dummy variable copies to avoid capturing `this` in kernel lambda
			           num_t t = m_time;
			           size_t n_bodies = m_n_bodies;
			           integrator_t integrator = m_integrator;

			           // Launch different kernel depending on the force choice
			           switch (m_force) {
			           case force_t::GRAVITY: {
				           // Again, dummy copies
				           num_t G = m_grav_params.G;
				           num_t damping = m_grav_params.damping;

				           cgh.parallel_for<kernel<num_t, 0>>(
					           cl::sycl::range<1>(m_n_bodies), [=](cl::sycl::item<1> item) {
						                                           auto id = item.get_linear_id();

						                                           // Computes the gravitational acceleration on a body using the
						                                           // chosen constants
						                                           const auto grav = [&](vec3<num_t>, vec3<num_t> x,
						                                                                 num_t) -> vec3<num_t> {
							                                           vec3<num_t> acc(0);

							                                           for (size_t i = 0; i < n_bodies; i++) {
								                                           auto const diff = pos[i] - x;
								                                           auto const r =
								                                           sycl::sqrt(diff.x() * diff.x() + diff.y() * diff.y() +
								                                                      diff.z() * diff.z());
								                                           acc += diff /
								                                                  (r * r * r + num_t(1e24) * num_t(i == id) + damping);
							                                           }

							                                           return G * acc;
						                                           };

						                                           // Use the chosen integrator to find new values of position and
						                                           // velocity
						                                           if (integrator == integrator_t::EULER) {
							                                           std::tie(wvel[id], wpos[id], std::ignore) =
							                                           integrate_step_euler(grav, STEP_SIZE, vel[id], pos[id],
							                                                                t);
						                                           } else if (integrator == integrator_t::RK4) {
							                                           std::tie(wvel[id], wpos[id], std::ignore) =
							                                           integrate_step_rk4(grav, STEP_SIZE, vel[id], pos[id], t);
						                                           }
					                                           });
			           } break;
			           case force_t::LENNARD_JONES: {
				           // Dummy copies
				           num_t eps = m_lj_params.eps;
				           num_t sigma = m_lj_params.sigma;
				           auto A = num_t(24) * eps * sigma;

				           cgh.parallel_for<kernel<num_t, 1>>(
					           cl::sycl::range<1>(m_n_bodies), [=](cl::sycl::item<1> item) {
						                                           auto id = item.get_linear_id();

						                                           // Computes the acceleration on a body from the sum of
						                                           // Lennard-Jones
						                                           // potentials between itself and all other bodies using the
						                                           // provided
						                                           // parameters
						                                           const auto force = [&](vec3<num_t>, vec3<num_t> x,
						                                                                  num_t) -> vec3<num_t> {
							                                           vec3<num_t> acc(0);

							                                           for (size_t i = 0; i < n_bodies; i++) {
								                                           auto const diff = pos[i] - x;
								                                           auto const r =
								                                           sycl::sqrt(diff.x() * diff.x() + diff.y() * diff.y() +
								                                                      diff.z() * diff.z()) +
								                                           num_t(1e24) * num_t(i == id);

								                                           acc += sycl::pow(r, num_t(-8)) * diff -
								                                                  num_t(2) * sycl::pow(r, num_t(-14)) * diff;
							                                           }

							                                           return A * acc;
						                                           };

						                                           // Use the chosen integrator to find new values of position and
						                                           // velocity
						                                           if (integrator == integrator_t::EULER) {
							                                           std::tie(wvel[id], wpos[id], std::ignore) =
							                                           integrate_step_euler(force, STEP_SIZE, vel[id], pos[id],
							                                                                t);
						                                           } else if (integrator == integrator_t::RK4) {
							                                           std::tie(wvel[id], wpos[id], std::ignore) =
							                                           integrate_step_rk4(force, STEP_SIZE, vel[id], pos[id], t);
						                                           }
					                                           });
			           } break;
			           case force_t::COULOMB: {
				           if (!m_coulomb_charges_buf) {
					           throw std::runtime_error("Coulomb charge buffer wasn't initialized!");
				           }

				           // Read accessor for particle charges
				           auto charges_acc = std::get<0>(m_coulomb_charges_buf->gen_read_accs(cgh, read_bufs_t<0>{}));

				           cgh.parallel_for<kernel<num_t, 2>>(
					           cl::sycl::range<1>(m_n_bodies), [=](cl::sycl::item<1> item) {
						                                               auto id = item.get_linear_id();
						                                               num_t my_charge = charges_acc[id];

						                                               // Computes the gravitational acceleration on a body using the
						                                               // chosen constants
						                                               const auto cmb = [&](vec3<num_t>, vec3<num_t> x,
						                                                                    num_t) -> vec3<num_t> {
							                                               vec3<num_t> acc(0);

							                                               for (size_t i = 0; i < n_bodies; i++) {
								                                               auto const diff = pos[i] - x;
								                                               auto const r =
								                                               sycl::sqrt(diff.x() * diff.x() + diff.y() * diff.y() +
								                                                          diff.z() * diff.z());
								                                               acc += charges_acc[i] * diff /
								                                                      (r * r * r + num_t(1e24) * num_t(i == id));
							                                               }

							                                               return my_charge * acc;
						                                               };

						                                               // Use the chosen integrator to find new values of position and
						                                               // velocity
						                                               if (integrator == integrator_t::EULER) {
							                                               std::tie(wvel[id], wpos[id], std::ignore) =
							                                               integrate_step_euler(cmb, STEP_SIZE, vel[id], pos[id],
							                                                                    t);
						                                               } else if (integrator == integrator_t::RK4) {
							                                               std::tie(wvel[id], wpos[id], std::ignore) =
							                                               integrate_step_rk4(cmb, STEP_SIZE, vel[id], pos[id], t);
						                                               }
					                                               });

			           } break;
			           }
		           });

		m_bufs.swap();
		m_time += STEP_SIZE;
	}
};
