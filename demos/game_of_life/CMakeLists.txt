add_executable(GameOfLife WIN32 main.cpp
                                sim.cpp)

link_demo_dependencies(TARGET GameOfLife)

add_sycl_to_target(TARGET GameOfLife
                   SOURCES sim.cpp)
