# SYCL Demos
This folder contains graphical demos using SYCL for acceleration.

## Dependencies
---
The demos use the following libraries:
* [Cinder](https://www.libcinder.org/)

### Game of Life
This demo simulates Conway's Game of Life with a dynamically resizable grid.
To draw new cells, hold the mouse button and drag the mouse slowly over the
grid. Press SPACE to pause/resume the simulation. To resize the grid, use the
mouse wheel. Doing this or resizing the window will reset the simulation.

### Mandelbrot
This demo dynamically renders and displays a visualization of the Mandelbrot
set on the complex plane. Use the mouse wheel to zoom in or out and drag the
mouse while holding the mouse button to move around the plane.

### NBody
This demo demonstrates the use of numerical integration methods to simulate
systems of interacting bodies, where every body exerts a force on every other
body. A graphical interace is provided to set the force type, the integration
method, and the initial distribution of bodies. The simulation can be
initialized from there. The simulation can be viewed from different positions
by dragging the mouse and using the mouse wheel to control the camera.

### Building the demos
To build the demos, call CMake with `-DCOMPUTECPP_SDK_BUILD_DEMOS=ON`

The following packages are required when building the demos on Linux
- libcurl4-openssl-dev
- mesa-common-dev
