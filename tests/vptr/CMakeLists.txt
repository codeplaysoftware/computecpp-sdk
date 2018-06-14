
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(basic basic.cc)
target_link_libraries(basic PUBLIC gtest
                            PUBLIC gtest_main
                            PUBLIC ${CMAKE_THREAD_LIBS_INIT})
add_dependencies(basic gtest_main)
add_dependencies(basic gtest)
add_sycl_to_target(
  TARGET basic
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/basic.cc
)
add_test(BasicTests basic)

add_executable(offset offset.cc)
target_link_libraries(offset PUBLIC gtest
                             PUBLIC gtest_main
                             PUBLIC ${CMAKE_THREAD_LIBS_INIT})
add_dependencies(offset gtest_main)
add_dependencies(offset gtest)
add_sycl_to_target(
  TARGET offset
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/offset.cc
)
add_test(OffsetTests offset)

if (COMPUTECPP_INTERFACE)
  add_definitions(-DCOMPUTECPP_INTERFACE=${COMPUTECPP_INTERFACE})
  add_executable(runtime runtime.cc)
  target_link_libraries(runtime PUBLIC gtest
                                PUBLIC gtest_main
                                PUBLIC ${CMAKE_THREAD_LIBS_INIT})
  add_dependencies(runtime gtest_main)
  add_dependencies(runtime gtest)
  add_sycl_to_target(
    TARGET runtime
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/runtime.cc
  )
  add_test(RuntimeTests runtime)
endif()


add_executable(space space.cc)
target_link_libraries(space PUBLIC gtest
                            PUBLIC gtest_main
                            PUBLIC ${CMAKE_THREAD_LIBS_INIT})
add_dependencies(space gtest_main)
add_dependencies(space gtest)
add_sycl_to_target(
  TARGET space
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/space.cc
)
add_test(SpaceTests space)

add_executable(accessor accessor.cc)
target_link_libraries(accessor PUBLIC gtest
                               PUBLIC gtest_main
                               PUBLIC ${CMAKE_THREAD_LIBS_INIT})
add_dependencies(accessor gtest_main)
add_dependencies(accessor gtest)
add_sycl_to_target(
  TARGET accessor
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/accessor.cc
)
add_test(AccessorTests accessor)

set_target_properties(basic offset space accessor PROPERTIES CXX_STANDARD 11)
