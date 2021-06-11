#pragma once

#include <iostream> // std::cout, std::endl
#include <vector> // std::vector
#include <cstdlib> // std::size_t
#include <algorithm> // std::fill
#include <exception> // std::exception_ptr
#include <cmath> // std::round

#include <CL/sycl.hpp>

// Kernel declarations.
class fluid_boundary;
class fluid_linear_solve;
class fluid_project1;
class fluid_project2;
class fluid_advect;
class image_kernal;

class SYCLFluidContainer {
public:
	SYCLFluidContainer(std::size_t size, float dt, float diffusion, float viscosity) :
		size{ size }, dt{ dt }, diffusion{ diffusion }, viscosity{ viscosity },
		// Create an image buffer.
		img{ cl::sycl::range<1>(size * size) },
		// Initialize queue with default selector and asynchronous exception handler.
		queue{cl::sycl::default_selector{}, [](cl::sycl::exception_list exceptions) {
				for (const std::exception_ptr& e : exceptions) {
					try {
						std::rethrow_exception(e);
					} catch (const cl::sycl::exception& e) {
						std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
					}
				}
			}
		} {
		auto s{ size * size };
		// Cache constant values for fluid math.
		a_velocity = dt * viscosity * (size - 2) * (size - 2);
		c_reciprocal_velocity = 1.0f / (1.0f + 6.0f * a_velocity);
		a_density = dt * diffusion * (size - 2) * (size - 2);
		c_reciprocal_density = 1.0f / (1.0f + 6.0f * a_density);
		dt0 = dt * size;
		// Resize fluid property vectors.
		px.resize(s);
		py.resize(s);
		x.resize(s);
		y.resize(s);
		previous_density.resize(s);
		density.resize(s);
	}

	~SYCLFluidContainer() = default;

	// Returns a pointer to the pixel data buffer.
	template <typename Func>
	void WithData(Func&& func) {
		auto acc{ img.template get_access<cl::sycl::access::mode::read>() };
		func(acc.get_pointer());
	}

	// Reset fluid to empty.
	void Reset() {
		std::fill(px.begin(), px.end(), 0.0f);
		std::fill(py.begin(), py.end(), 0.0f);
		std::fill(x.begin(), x.end(), 0.0f);
		std::fill(y.begin(), y.end(), 0.0f);
		std::fill(previous_density.begin(), previous_density.end(), 0.0f);
		std::fill(density.begin(), density.end(), 0.0f);
	}

	// Fade density over time.
	void DecreaseDensity(float fraction = 0.99f) {
		for (auto& d : density) {
			d *= fraction;
		}
	}

	// Add density to the density field.
	void AddDensity(std::size_t x, std::size_t y, float amount, int radius = 0) {
		if (radius > 0) {
			// Add density in circle around cursor.
			for (int i{ -radius }; i <= radius; ++i) {
				for (int j{ -radius }; j <= radius; ++j) {
					if (i * i + j * j <= radius * radius) {
						auto index{IX(x + i, y + j, size)};
						assert(index < this->density.size());
						this->density[index] += amount;
					}
				}
			}
		} else {
			// Add density at cursor location.
			auto index{ IX(x, y, size) };
			assert(index < this->density.size());
			this->density[index] += amount;
		}
	}

	// Add velocity to the velocity field.
	void AddVelocity(std::size_t x, std::size_t y, float px, float py) {
		auto index{IX(x, y, size)};
		assert(index < this->x.size());
		assert(index < this->y.size());
		this->x[index] += px;
		this->y[index] += py;
	}

	// Update function defined in fluid.cpp for SYCL integration header to be generated.
	void Update();

	// Updates the physics of the fluid.
	void UpdateImpl() {
		// Create a bunch of temporary buffers.
		float_buffer x_b{ x.data(), x.size(), props };
		float_buffer y_b{ y.data(), y.size(), props };
		float_buffer px_b{ px.data(), px.size(), props };
		float_buffer py_b{ py.data(), py.size(), props };
		float_buffer previous_density_b{ previous_density.data(), previous_density.size(), props };
		float_buffer density_b{ density.data(), density.size(), props };

		// Diffuse the fluid velocities.
		for (std::size_t iteration{0}; iteration < velocity_iterations; ++iteration) {
			Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto px_a) {
				LinearSolve(1, px_a, x_a, a_velocity, c_reciprocal_velocity, size, cgh);
			}, x_b, px_b);
			Submit(queue, [&](cl::sycl::handler& cgh, auto y_a, auto py_a) {
				LinearSolve(2, py_a, y_a, a_velocity, c_reciprocal_velocity, size, cgh);
			}, y_b, py_b);
			Submit(queue, [&](cl::sycl::handler& cgh, auto px_a) {
				SetBoundaryConditions(1, px_a, size, cgh);
			}, px_b);
			Submit(queue, [&](cl::sycl::handler& cgh, auto py_a) {
				SetBoundaryConditions(2, py_a, size, cgh);
			}, py_b);
		}

		// Project and advect the fluid velocities.
		Project(px_b, py_b, x_b, y_b);
		Advect(1, x_b, px_b, px_b, py_b);
		Advect(2, y_b, py_b, px_b, py_b);
		Project(x_b, y_b, px_b, py_b);

		// Diffuse the fluid densities.
		for (std::size_t iteration{0}; iteration < density_iterations; ++iteration) {
			Submit(queue, [&](cl::sycl::handler& cgh, auto previous_density_a, auto density_a) {
				LinearSolve(0, previous_density_a, density_a, a_density, c_reciprocal_density, size, cgh);
			}, previous_density_b, density_b);
			Submit(queue, [&](cl::sycl::handler& cgh, auto previous_density_a) {
				SetBoundaryConditions(0, previous_density_a, size, cgh);
			}, previous_density_b);
		}

		// Advect the fluid densities.
		Advect(0, density_b, previous_density_b, x_b, y_b);
		  
		// Update the image pixel data with the appropriate color for a given density.
		queue.submit([&](cl::sycl::handler& cgh) {
			auto img_acc{ img.template get_access<cl::sycl::access::mode::discard_write>(cgh) };
			auto density_a{ density_b.template get_access<cl::sycl::access::mode::read>(cgh) };
			cgh.parallel_for<image_kernal>(cl::sycl::range<1>(size * size), [=](cl::sycl::item<1> item) {
				auto index{ item.get_id(0) };
				auto value{ density_a[index] };
				std::uint8_t red = value >= 255 ? 255 : static_cast<std::uint8_t>(value);
				img_acc[index] = { red, 0, 0, 255 };
			});
		});

		queue.wait();
	}

	// Some aliases to improve readability of code.
	using float_buffer = cl::sycl::buffer<float, 1>;
	using read_write_accessor = cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t>;

	// Wrapper around queue submission.
	template <typename T, typename ...Ts>
	static void Submit(cl::sycl::queue& queue, T lambda, Ts&... buffers) {
		queue.submit([&](cl::sycl::handler& cgh) {
			lambda(cgh, CreateAccessor(cgh, buffers)...);
		});
	}

	// Creates read_write accessors from a buffer.
	template <typename T>
	static read_write_accessor CreateAccessor(cl::sycl::handler& cgh, T buffer) {
		return buffer.template get_access<cl::sycl::access::mode::read_write>(cgh);
	}

	// Clamp value to a range.
	template <typename T>
	static T Clamp(T value, T low, T high) {
		return value >= high ? high : value <= low ? low : value;
	}

	// Get clamped index based off of coordinates.
	static std::size_t IX(std::size_t x, std::size_t y, std::size_t N) {
		// Clamp coordinates.
		if (x < 0) { x = 0; }
		if (x > N - 1) { x = N - 1; }
		if (y < 0) { y = 0; }
		if (y > N - 1) { y = N - 1; }

		return (y * N) + x;
	}

	// Set boundaries to opposite of adjacent layer. (SYCL VERSION).
	static void SetBoundaryConditions(int b, read_write_accessor x, std::size_t N, cl::sycl::handler& cgh) {
		cgh.single_task<fluid_boundary>([=]() {
			for (std::size_t i{ 1 }; i < N - 1; ++i) {
				auto top{ IX(i, 1, N) };
				auto bottom{ IX(i, N - 2, N) };
				x[IX(i, 0, N)] = b == 2 ? -x[top] : x[top];
				x[IX(i, N - 1, N)] = b == 2 ? -x[bottom] : x[bottom];
			}

			for (std::size_t j{ 1 }; j < N - 1; ++j) {
				auto left{ IX(1, j, N) };
				auto right{ IX(N - 2, j, N) };
				x[IX(0, j, N)] = b == 1 ? -x[left] : x[left];
				x[IX(N - 1, j, N)] = b == 1 ? -x[right] : x[right];
			}

			// Set corner boundaries
			x[IX(0, 0, N)] = 0.33f * (x[IX(1, 0, N)] + x[IX(0, 1, N)] + x[IX(0, 0, N)]);
			x[IX(0, N - 1, N)] = 0.33f * (x[IX(1, N - 1, N)] + x[IX(0, N - 2, N)] + x[IX(0, N - 1, N)]);
			x[IX(N - 1, 0, N)] = 0.33f * (x[IX(N - 2, 0, N)] + x[IX(N - 1, 1, N)] + x[IX(N - 1, 0, N)]);
			x[IX(N - 1, N - 1, N)] = 0.33f * (x[IX(N - 2, N - 1, N)] + x[IX(N - 1, N - 2, N)] + x[IX(N - 1, N - 1, N)]);
		});
	}

	// Solve linear differential equation of density / velocity. (SYCL VERSION).
	static void LinearSolve(int b, read_write_accessor x, read_write_accessor x0, float a, float c_reciprocal, std::size_t N, cl::sycl::handler& cgh) {
		cgh.parallel_for<fluid_linear_solve>(cl::sycl::range<2>(N - 2, N - 2), [=](cl::sycl::item<2> item) {
			auto i{ 1 + item.get_id(0) };
			auto j{ 1 + item.get_id(1) };
			auto index{ IX(i, j, N) };
			x[index] = (x0[index] +
						a * (
							x[IX(i + 1, j, N)]
							+ x[IX(i - 1, j, N)]
							+ x[IX(i, j + 1, N)]
							+ x[IX(i, j - 1, N)]
							+ x[index]
							+ x[index]
							)
						) * c_reciprocal;
		});
	}

	// Converse 'mass' of density / velocity fields. (SYCL VERSION part 1).
	static void Project1(read_write_accessor vx, read_write_accessor vy, read_write_accessor p, read_write_accessor div, std::size_t N, cl::sycl::handler& cgh) {
		cgh.parallel_for<fluid_project1>(cl::sycl::range<2>(N - 2, N - 2), [=](cl::sycl::item<2> item) {
			auto i{ 1 + item.get_id(0) };
			auto j{ 1 + item.get_id(1) };
			auto index{ IX(i, j, N) };
			div[index] = -0.5f * (
				vx[IX(i + 1, j, N)]
				- vx[IX(i - 1, j, N)]
				+ vy[IX(i, j + 1, N)]
				- vy[IX(i, j - 1, N)]
				) / N;
			p[index] = 0;
		});
	}

	// Converse 'mass' of density / velocity fields. (SYCL VERSION part 2).
	static void Project2(read_write_accessor vx, read_write_accessor vy, read_write_accessor p, std::size_t N, cl::sycl::handler& cgh) {
		cgh.parallel_for<fluid_project2>(cl::sycl::range<2>(N - 2, N - 2), [=](cl::sycl::item<2> item) {
			auto i{ 1 + item.get_id(0) };
			auto j{ 1 + item.get_id(1) };
			auto index{ IX(i, j, N) };
			vx[index] -= 0.5f * (p[IX(i + 1, j, N)] - p[IX(i - 1, j, N)]) * N;
			vy[index] -= 0.5f * (p[IX(i, j + 1, N)] - p[IX(i, j - 1, N)]) * N;
		});
	}

	void Project(float_buffer& px_b, float_buffer& py_b, float_buffer& x_b, float_buffer& y_b) {
		Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto px_a, auto y_a, auto py_a) {
			Project1(px_a, py_a, x_a, y_a, size, cgh);
		}, x_b, px_b, y_b, py_b);

		Submit(queue, [&](cl::sycl::handler& cgh, auto x_a) {
			SetBoundaryConditions(0, x_a, size, cgh);
		}, x_b);
		Submit(queue, [&](cl::sycl::handler& cgh, auto y_a) {
			SetBoundaryConditions(0, y_a, size, cgh);
		}, y_b);

		for (std::size_t iteration{ 0 }; iteration < velocity_iterations; ++iteration) {
			Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto y_a) {
				LinearSolve(0, x_a, y_a, 1.0f, c_reciprocal_project, size, cgh);
			}, x_b, y_b);
			Submit(queue, [&](cl::sycl::handler& cgh, auto x_a) {
				SetBoundaryConditions(0, x_a, size, cgh);
			}, x_b);
		}

		Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto px_a, auto py_a) {
			Project2(px_a, py_a, x_a, size, cgh);
		}, x_b, px_b, py_b);
		Submit(queue, [&](cl::sycl::handler& cgh, auto px_a) {
			SetBoundaryConditions(1, px_a, size, cgh);
		}, px_b);
		Submit(queue, [&](cl::sycl::handler& cgh, auto py_a) {
			SetBoundaryConditions(2, py_a, size, cgh);
		}, py_b);
	}

	// Move density / velocity within the field to the next step. (SYCL VERSION).
	static void AdvectImpl(int b, read_write_accessor d, read_write_accessor d0, read_write_accessor u, read_write_accessor v, float dt0, std::size_t N, cl::sycl::handler& cgh) {
		cgh.parallel_for<fluid_advect>(cl::sycl::range<2>(N - 2, N - 2), [=](cl::sycl::item<2> item) {
			auto i{ 1 + item.get_id(0) };
			auto j{ 1 + item.get_id(1) };
			auto index{ IX(i, j, N) };
			float x{ i - dt0 * u[index] };
			float y{ j - dt0 * v[index] };
			x = Clamp(x, 0.5f, N + 0.5f);
			auto i0{ (int)x };
			auto i1{ i0 + 1 };
			y = Clamp(y, 0.5f, N + 0.5f);
			auto j0{ (int)y };
			auto j1{ j0 + 1 };
			float s1{ x - i0 };
			float s0{ 1 - s1 };
			float t1{ y - j0 };
			float t0{ 1 - t1 };
			d[index] = s0 * (t0 * d0[IX(i0, j0, N)] + t1 * d0[IX(i0, j1, N)]) +
				s1 * (t0 * d0[IX(i1, j0, N)] + t1 * d0[IX(i1, j1, N)]);
		});
	}

	void Advect(int b, float_buffer& d, float_buffer& d0, float_buffer& u, float_buffer& v) {
		Submit(queue, [&](cl::sycl::handler& cgh, auto d_a, auto d0_a, auto u_a, auto v_a) {
			AdvectImpl(b, d_a, d0_a, u_a, v_a, dt0, size, cgh);
		}, d, d0, u, v);
		Submit(queue, [&](cl::sycl::handler& cgh, auto d_a) {
			SetBoundaryConditions(b, d_a, size, cgh);
		}, d);
	}

	// Edge length of fluid container (always square).
	std::size_t size{ 0 };

	std::size_t velocity_iterations{ 4 };
	std::size_t density_iterations{ 4 };

	// Internal constants for fluid math.
	float dt{ 0.0f };
	float diffusion{ 0.0f };
	float viscosity{ 0.0f };
	float a_velocity{ 0.0f };
	float a_density{ 0.0f };
	float c_reciprocal_velocity{ 0.0f };
	float c_reciprocal_density{ 0.0f };
	float c_reciprocal_project{ 1.0f / 6.0f };
	float dt0{ 0.0f };

	// Previous velocity components.
	std::vector<float> px;
	std::vector<float> py;
	
	// Current velocity components.
	std::vector<float> x;
	std::vector<float> y;

	std::vector<float> previous_density;
	std::vector<float> density;

	// SYCL objects.
	cl::sycl::property_list props{ cl::sycl::property::buffer::use_host_ptr() };
	cl::sycl::queue queue;
	cl::sycl::buffer<cl::sycl::cl_uchar4, 1> img;
};