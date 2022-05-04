cmake_minimum_required(VERSION 3.10.2)

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in
               ${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build)
include_directories(${gtest_SOURCE_DIR}/include)

find_package(Threads REQUIRED)

function(ptr_test)
  set(options)
  set(one_value_args TARGET)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(ARG
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN})
  add_executable(${ARG_TARGET} ${ARG_SOURCES})
  set_property(TARGET ${ARG_TARGET} PROPERTY CXX_STANDARD 14)
  target_include_directories(${ARG_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(${ARG_TARGET} PUBLIC gtest gtest_main Threads::Threads)
  add_sycl_to_target(TARGET ${ARG_TARGET} SOURCES ${ARG_SOURCES})
  add_test(NAME ${ARG_TARGET} COMMAND ${ARG_TARGET})
  install(TARGETS ${ARG_TARGET} RUNTIME DESTINATION bin)
endfunction(ptr_test)

add_subdirectory(legacy-pointer)
add_subdirectory(vptr)
