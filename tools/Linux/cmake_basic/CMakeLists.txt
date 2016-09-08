project(ComputeCpp-samples-sdk)

cmake_minimum_required(VERSION 3.2)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules)

include(FindComputeCpp)

include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

add_executable(MYFILE ${CMAKE_CURRENT_SOURCE_DIR}/parallel_for.cpp)
target_compile_options(MYFILE PUBLIC -std=c++11 -Wall)
add_sycl_to_target(MYFILE ${CMAKE_CURRENT_SOURCE_DIR}/parallel_for.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR})
