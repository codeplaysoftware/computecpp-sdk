project (nbody)

cmake_minimum_required(VERSION 3.2.2)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake/Modules")
include(FindOpenCL)
include(FindComputeCpp)

set(CMAKE_CXX_STANDARD 11)

include("${CINDER_PATH}/proj/cmake/modules/cinderMakeApp.cmake")

include_directories(SYSTEM "${COMPUTECPP_INCLUDE_DIRECTORY}")
include_directories(SYSTEM "${OpenCL_INCLUDE_DIR}")
include_directories(SYSTEM "${CINDER_PATH}/include")
include_directories("${CMAKE_SOURCE_DIR}/include")

# Setup ImGui for Cinder
include_directories("${CIGUI_PATH}/include")
include_directories("${CIGUI_PATH}/lib/imgui")

set(CIGUI_SOURCES
    "${CIGUI_PATH}/src/CinderImGui.cpp"
    "${CIGUI_PATH}/lib/imgui/imgui.cpp"
    "${CIGUI_PATH}/lib/imgui/imgui_draw.cpp"
    "${CIGUI_PATH}/lib/imgui/imgui_demo.cpp")

add_library(cinder_imgui "${CIGUI_SOURCES}")
target_link_libraries(cinder_imgui cinder)

add_subdirectory(game_of_life)
add_subdirectory(mandelbrot)
add_subdirectory(nbody)
add_subdirectory(domainwarping)
