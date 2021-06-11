add_executable(FluidSimulation WIN32 main.cpp
                                     fluid.cpp)

link_demo_dependencies(TARGET FluidSimulation)

add_sycl_to_target(TARGET FluidSimulation
                   SOURCES fluid.cpp)