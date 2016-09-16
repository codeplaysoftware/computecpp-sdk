project(ComputeCpp-samples-sdk)

# Set minimum required version of CMake to 3.2.2
cmake_minimum_required(VERSION 3.2.2)

# Include modules
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake/Modules)
include(FindOpenCL)
include(FindComputeCpp)

# Set include ComputeCpp directories
include_directories(SYSTEM ${COMPUTECPP_INCLUDE_DIRECTORY})
include_directories(SYSTEM ${OpenCL_INCLUDE_DIR})

# warnings for SDK software
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Enable testing
enable_testing()

# Add subdirectories for samples
file(GLOB _ALL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_FILE ${_ALL_FILES})
  if (IS_DIRECTORY ${_FILE})
    if (EXISTS ${_FILE}/CMakeLists.txt)
     add_subdirectory(${_FILE})
    endif()
  endif()
endforeach()
