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

include(FindComputeCpp)

include_directories(${gtest_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
message(STATUS "Include dirs: ${gtest_SOURCE_DIR}")

add_executable(basic basic.cc)
set_property(TARGET basic PROPERTY CXX_STANDARD 11)
message(STATUS "LIBRARY ${COMPUTECPP_LIBRARY}")

target_link_libraries(basic PUBLIC ${gtest_BINARY_DIR}/libgtest.a)
target_link_libraries(basic PUBLIC ${gtest_BINARY_DIR}/libgtest_main.a)
target_link_libraries(basic PUBLIC pthread)
add_dependencies(basic gtest_main)
add_dependencies(basic gtest)
add_sycl_to_target(basic ${CMAKE_CURRENT_SOURCE_DIR}/basic.cc ${CMAKE_CURRENT_BINARY_DIR})
add_test(BasicTests basic)

add_executable(offset offset.cc)
set_property(TARGET basic PROPERTY CXX_STANDARD 11)
target_link_libraries(offset PUBLIC ${gtest_BINARY_DIR}/libgtest.a)
target_link_libraries(offset PUBLIC ${gtest_BINARY_DIR}/libgtest_main.a)
target_link_libraries(offset PUBLIC pthread)
add_dependencies(offset gtest_main)
add_dependencies(offset gtest)
add_sycl_to_target(offset ${CMAKE_CURRENT_SOURCE_DIR}/offset.cc ${CMAKE_CURRENT_BINARY_DIR})
add_test(OffsetTests offset)
