#include <cinder/CameraUi.h>
#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>

#include <CL/sycl.hpp>

#include "fluid.h"

// Size of the fluid container edge (always square shaped).
const int SIZE{ 300 };

// Default scale at which fluid container is rendered.
const int SCALE{ 3 };

class FluidSimulationApp : public ci::app::App {
public:
	FluidSimulationApp() : size_{ SIZE }, fluid_{ SIZE, 0.2f, 0.0f, 0.0000001f } {
		// Set window size.
		getWindow()->setSize({ size_ * SCALE, size_ * SCALE });
		// Add usage instruction to application title
		getWindow()->setTitle("Fluid Simulation - Move mouse to add fluid - Press space to clear fluid");
	}

	// Create fluid texture.
	void setup() override {
		texture_ = ci::gl::Texture2d::create(
			nullptr, 
			GL_RGBA, 
			size_, 
			size_,
			ci::gl::Texture2d::Format().dataType(GL_UNSIGNED_BYTE).internalFormat(GL_RGBA)
		);
	}

	// Called once per frame to update the fluid.
	void update() override {
		// Check that mouse has moved.
		if (prev_x != 0 && prev_x != 0) {
			// Add density at mouse cursor location.
			auto x{ static_cast<std::size_t>(prev_x * size_) };
			auto y{ static_cast<std::size_t>(prev_y * size_) };
			fluid_.AddDensity(x, y, 400, 2);
		}
		
		// Fade overall dye levels slowly over time.
		fluid_.DecreaseDensity(0.99f);
		
		// Update fluid physics.
		fluid_.Update();
	}

	// Draws fluid to the screen.
	void draw() override {
		// Clear screen.
		ci::gl::clear();

		// Update texture with pixel data array.
		fluid_.WithData([&](cl::sycl::cl_uchar4 const* data) {
			texture_->update(data, GL_RGBA, GL_UNSIGNED_BYTE, 0, size_, size_);
		});

		// Draw texture to screen.
		ci::Rectf screen(0.0f, 0.0f, static_cast<float>(getWindow()->getWidth()), static_cast<float>(getWindow()->getHeight()));
		ci::gl::draw(texture_, screen);
	}
private:
	void keyDown(ci::app::KeyEvent event) override {
		// Reset fluid container to empty if SPACE key is pressed.
		if (event.getCode() == ci::app::KeyEvent::KEY_SPACE) {
			fluid_.Reset();
		}
	}

	// Dragging mouse does not include moving mouse so defined separately here.
	void mouseDrag(ci::app::MouseEvent event) override {
		mouseMove(event);
	}

	void mouseMove(ci::app::MouseEvent event) override {
		// Get mouse position as fraction of window size.
		auto x{ event.getX() / float(getWindow()->getSize().x) };
		auto y{ 1.0f - event.getY() / float(getWindow()->getSize().y) };
		// Check that previous mouse position was not only zero (for initial value to be set).
		if (prev_x != 0.0 || prev_y != 0.0) {
			// Get amount by which mouse has moved.
			auto amount_x{ (x - prev_x) * size_ };
			auto amount_y{ (y - prev_y) * size_ };
			// Add velocity and density in direction of mouse travel.
			// Multiplied by size because x and y are in 0.0 to 1.0 relative window coordinates.
			auto current_x{ static_cast<std::size_t>(x * size_) };
			auto current_y{ static_cast<std::size_t>(y * size_) };
			fluid_.AddVelocity(current_x, current_y, amount_x, amount_y);
		}
		// Update previous mouse position.
		prev_x = x;
		prev_y = y;
	}
	
	// Square size of fluid container.
	int size_{ 0 };
	
	// Store previous mouse positions.
	float prev_x{ 0 };
	float prev_y{ 0 };

	// Fluid container object.
	SYCLFluidContainer fluid_;

	// Fluid texture.
	ci::gl::Texture2dRef texture_;
};

// Cinder app initialization.
CINDER_APP(FluidSimulationApp, ci::app::RendererGl(ci::app::RendererGl::Options{}));