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

include(FindOpenCL)
include(FindComputeCpp)


include_directories(${COMPUTECPP_INCLUDE_DIRECTORY}) 
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
add_sycl_to_target(basic  ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/basic.cc)
add_test(BasicTests basic)

add_executable(offset offset.cc)
set_property(TARGET offset PROPERTY CXX_STANDARD 11)
target_link_libraries(offset PUBLIC ${gtest_BINARY_DIR}/libgtest.a)
target_link_libraries(offset PUBLIC ${gtest_BINARY_DIR}/libgtest_main.a)
target_link_libraries(offset PUBLIC pthread)
add_dependencies(offset gtest_main)
add_dependencies(offset gtest)
add_sycl_to_target(offset  ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/offset.cc)
add_test(OffsetTests offset)

if (COMPUTECPP_INTERFACE)
  add_executable(runtime runtime.cc)
  set_property(TARGET runtime PROPERTY CXX_STANDARD 11)
  target_link_libraries(runtime PUBLIC ${gtest_BINARY_DIR}/libgtest.a)
  target_link_libraries(runtime PUBLIC ${gtest_BINARY_DIR}/libgtest_main.a)
  target_link_libraries(runtime PUBLIC pthread)
  add_dependencies(runtime gtest_main)
  add_dependencies(runtime gtest)
  add_sycl_to_target(runtime  ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/runtime.cc)
  add_test(RuntimeTests runtime)
endif()


add_executable(space space.cc)
set_property(TARGET space PROPERTY CXX_STANDARD 11)
target_link_libraries(space PUBLIC ${gtest_BINARY_DIR}/libgtest.a)
target_link_libraries(space PUBLIC ${gtest_BINARY_DIR}/libgtest_main.a)
target_link_libraries(space PUBLIC pthread)
add_dependencies(space gtest_main)
add_dependencies(space gtest)
add_sycl_to_target(space  ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/space.cc)
add_test(SpaceTests space)

add_executable(accessor accessor.cc)
set_property(TARGET accessor PROPERTY CXX_STANDARD 11)
target_link_libraries(accessor PUBLIC ${gtest_BINARY_DIR}/libgtest.a)
target_link_libraries(accessor PUBLIC ${gtest_BINARY_DIR}/libgtest_main.a)
target_link_libraries(accessor PUBLIC pthread)
add_dependencies(accessor gtest_main)
add_dependencies(accessor gtest)
add_sycl_to_target(accessor  ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/accessor.cc)
add_test(AccessorTests accessor)

