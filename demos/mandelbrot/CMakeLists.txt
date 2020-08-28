add_executable(Mandelbrot WIN32 main.cpp
                                mandel.cpp)

link_demo_dependencies(TARGET Mandelbrot)

add_sycl_to_target(TARGET Mandelbrot
                   SOURCES mandel.cpp)
