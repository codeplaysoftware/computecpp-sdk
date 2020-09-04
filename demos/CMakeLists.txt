cmake_minimum_required(VERSION 3.12)

project(ComputeCpp-SDK-Demos)

# include(ExternalProject)

# ExternalProject_Add(
#     cinder_project
#     PREFIX          Cinder
#     GIT_REPOSITORY  https://github.com/cinder/Cinder.git
#     GIT_TAG         6eb7208682e92eadb168d59b310fe2637e934075
#     CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
#                     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#                     -DOpenGL_GL_PREFERENCE=GLVND
#                     -DCINDER_DISABLE_AUDIO=ON
#                     -DCINDER_DISABLE_VIDEO=ON
#     INSTALL_COMMAND ""
# )
# ExternalProject_Get_property(cinder_project SOURCE_DIR)
# set(CINDER_PATH "${SOURCE_DIR}")

# Creates a utility target used to trigger the download of the cinder source code
# add_custom_target(cinder DEPENDS cinder_project)

include(FetchContent)

FetchContent_Declare(
    CinderProject
    GIT_REPOSITORY https://github.com/cinder/Cinder.git
    GIT_TAG        6eb7208682e92eadb168d59b310fe2637e934075
)
FetchContent_MakeAvailable(CinderProject)

set_target_properties(
    cinder PROPERTIES
    CXX_STANDARD 17
)

set(DEMOS_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

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

  target_include_directories(${LINK_DEMO_TARGET} PRIVATE "${CINDER_PATH}/include"
                                                         "${DEMOS_ROOT_DIR}/include")

  set_target_properties(${LINK_DEMO_TARGET} PROPERTIES
                        CXX_STANDARD 17)
                        #MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

  target_include_directories(${LINK_DEMO_TARGET} PRIVATE "../include")
  target_link_libraries(${LINK_DEMO_TARGET} PRIVATE cinder)

  if (WIN32)
    if (${CMAKE_BUILD_TYPE} MATCHES Debug)
      set(_computecpp_dll "${ComputeCpp_DIR}/lib/ComputeCpp_d.dll")
    else()
      set(_computecpp_dll "${ComputeCpp_DIR}/lib/ComputeCpp.dll")
    endif()
    add_custom_command(TARGET ${LINK_DEMO_TARGET} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy ${_computecpp_dll} ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  # if (WIN32)
  #   target_link_directories(${LINK_DEMO_TARGET} PRIVATE ${CINDER_PATH}/lib/msw/x64/${CMAKE_BUILD_TYPE}/v${MSVC_TOOLSET_VERSION})
  #   target_link_libraries(${LINK_DEMO_TARGET} PRIVATE cinder.lib)
  #
  #   if (${CMAKE_BUILD_TYPE} MATCHES Debug)
  #     message(FATAL_ERROR "Building the demos in debug is not yet supported")
  #   else()
  #     # Used to copy the DLL to the demos folder so they can be executed directly
  #     set(COMPUTECPP_DLL "${ComputeCpp_DIR}/lib/ComputeCpp.dll")
  #   endif()
  #   add_custom_command(TARGET ${LINK_DEMO_TARGET} POST_BUILD
  #                      COMMAND ${CMAKE_COMMAND} -E copy ${COMPUTECPP_DLL} ${CMAKE_CURRENT_BINARY_DIR})
  #
  #   # Cinder is built as a static library on Windows, therefore we need to select the correct version
  #   # of the MSVC runtime libraries or linking will fail
  #   set_target_properties(${LINK_DEMO_TARGET} PROPERTIES
  #                           CXX_STANDARD 17
  #                           MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  # else()
  #   target_link_directories(${LINK_DEMO_TARGET} PRIVATE ${CINDER_PATH}/lib/linux/x86_64/ogl/${CMAKE_BUILD_TYPE})
  #
  #   # On Linux we need to manually link against cinder's dependencies to be able to finish the executable
  #   find_package(OpenGL REQUIRED)
  #   find_package(X11 REQUIRED)
  #   find_package(Threads REQUIRED)
  #   find_package(CURL REQUIRED)
  #   find_package(Fontconfig REQUIRED)
  #   target_link_libraries(${LINK_DEMO_TARGET} PRIVATE libcinder.a
  #                                                     -ldl
  #                                                     ${X11_LIBRARIES}
  #                                                     Xcursor Xinerama Xrandr Xi
  #                                                     Threads::Threads
  #                                                     OpenGL::OpenGL
  #                                                     CURL::libcurl
  #                                                     Fontconfig::Fontconfig)
  # endif()
endfunction(link_demo_dependencies)

# add_subdirectory(game_of_life)
# add_subdirectory(mandelbrot)
add_subdirectory(nbody)
