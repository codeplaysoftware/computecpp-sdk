cmake_minimum_required(VERSION 3.10.2)
project(ComputeCpp-samples-sdk)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package(ComputeCpp REQUIRED)

add_executable(MYFILE parallel_for.cpp)
target_compile_options(MYFILE PUBLIC -std=c++11 -Wall)
add_sycl_to_target(TARGET MYFILE SOURCES parallel_for.cpp)
