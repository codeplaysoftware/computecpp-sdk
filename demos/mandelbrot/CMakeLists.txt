add_executable(Mandelbrot WIN32 main.cpp
                                mandel-float.cpp
                                mandel-double.cpp)

link_demo_dependencies(TARGET Mandelbrot)

add_sycl_to_target(TARGET Mandelbrot
                   SOURCES mandel-float.cpp
                           mandel-double.cpp)
