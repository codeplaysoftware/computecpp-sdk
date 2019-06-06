# SYCL Demos
This folder contains graphical demos using SYCL for acceleration.

## Dependencies
The demos use the following libraries:

- [Cinder](https://www.libcinder.org/)
- [ImGui](https://github.com/ocornut/imgui/)
- [Cinder-ImGui](https://github.com/simongeilfus/Cinder-ImGui/)

These have to be built manually. First, please install the dependencies outlined
[here](https://libcinder.org/docs/branch/master/guides/linux-notes/ubuntu.html),
as well as libboost-system-dev and libboost-filesystem-dev on Ubuntu or their
equivalents on other distributions. Then, clone two repositories into a folder
of your choosing:

```shell
git clone --recursive https://github.com/cinder/Cinder
git clone --recursive https://github.com/simongeilfus/Cinder-ImGui
```

In the Cinder folder (after `cd Cinder`), execute the following:

```shell
rm -rf include/boost/
rm -rf lib/linux/x86_64/libboost_*
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCINDER_BOOST_USE_SYSTEM=1 ..
make
cmake -DCMAKE_BUILD_TYPE=Release -DCINDER_BOOST_USE_SYSTEM=1 ..
make
```

Then navigate to the `Cinder-ImGui` folder where you have to change the commit of the `imgui` dependency. \
After `cd lib/imgui`, checkout the working commit as follows:

```
git checkout df8a9c49eb6d9f134411eeffa0441f561aec3967
```

## Building
After building Cinder, you can build the demos by running these commands in the
`demos` folder of the SDK:

```shell
mkdir build
cd build
cmake -DComputeCpp_DIR=/path/to/ccp/package/ \
      -DCINDER_PATH=/path/to/cinder/repo/ \
      -DCIGUI_PATH=/path/to/cinder-imgui/repo/ \
      -DCOMPUTECPP_USER_FLAGS="-Xclang -cl-denorms-are-zero" \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make
```

## Running
The demo binaries will be built in the `build/Debug` or `build/Release`
folder and can be run from there.

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
