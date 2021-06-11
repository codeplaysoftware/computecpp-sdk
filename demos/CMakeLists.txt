cmake_minimum_required(VERSION 3.12)

include(FetchContent)

set(FETCHCONTENT_QUIET OFF CACHE STRING "Enable verbose Fetch Content" FORCE)

FetchContent_Declare(
    CinderProject
    GIT_REPOSITORY https://github.com/cinder/Cinder.git
    GIT_TAG        6eb7208682e92eadb168d59b310fe2637e934075
)

# Set options for a minimal cinder build. They need
# to be set before the call to FetchContent_MakeAvailable
# so they are propagated to Cinder
set(OpenGL_GL_PREFERENCE GLVND)
set(CINDER_DISABLE_AUDIO ON)
set(CINDER_DISABLE_VIDEO ON)

FetchContent_MakeAvailable(CinderProject)

if (WIN32)
  # Cinder requires C++17 to be built on Windows
  set_target_properties(cinder PROPERTIES CXX_STANDARD 17)
else()
  # But it requires -std=c++11 on Linux. The following properties
  # are required to make sure cmake passes -std=c++11 in the command line.
  set_target_properties(cinder PROPERTIES CXX_STANDARD_REQUIRED ON
                                          CXX_EXTENSIONS        OFF
                                          CXX_STANDARD          11)
endif()

set(_demos_root_dir ${CMAKE_CURRENT_SOURCE_DIR})

#############################
#   link_demo_dependencies
#############################
#
#  Utility function to link a demo to the required dependencies.
#
#  TARGET : Name of the target.
#
function(link_demo_dependencies)
  set(options)
  set(one_value_args
    TARGET
  )
  set(multi_value_args)
  cmake_parse_arguments(LINK_DEMO
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  set_target_properties(${LINK_DEMO_TARGET} PROPERTIES CXX_STANDARD 17)
  target_include_directories(${LINK_DEMO_TARGET} PRIVATE "${CINDER_PATH}/include"
                                                         "${_demos_root_dir}/include")
  target_link_libraries(${LINK_DEMO_TARGET} PRIVATE cinder)

  if (WIN32)
    # Copy the ComputeCpp.dll to target folder so demos can be executed directly
    if (${CMAKE_BUILD_TYPE} MATCHES Debug)
      set(_computecpp_dll "${ComputeCpp_DIR}/lib/ComputeCpp_d.dll")
    else()
      set(_computecpp_dll "${ComputeCpp_DIR}/lib/ComputeCpp.dll")
    endif()
    add_custom_command(TARGET ${LINK_DEMO_TARGET} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy ${_computecpp_dll} ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endfunction(link_demo_dependencies)

add_subdirectory(game_of_life)
add_subdirectory(mandelbrot)
add_subdirectory(nbody)
add_subdirectory(fluid)
