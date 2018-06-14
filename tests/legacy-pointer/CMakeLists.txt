include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(legacy_basic basic.cc)
target_link_libraries(legacy_basic PUBLIC gtest)
target_link_libraries(legacy_basic PUBLIC gtest_main)
target_link_libraries(legacy_basic PUBLIC ${CMAKE_THREAD_LIBS_INIT})
add_dependencies(legacy_basic gtest_main)
add_dependencies(legacy_basic gtest)
add_sycl_to_target(
  TARGET legacy_basic
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/basic.cc
)
add_test(BasicTests legacy_basic)

add_executable(legacy_offset offset.cc)
target_link_libraries(legacy_offset PUBLIC gtest)
target_link_libraries(legacy_offset PUBLIC gtest_main)
target_link_libraries(legacy_offset PUBLIC ${CMAKE_THREAD_LIBS_INIT})
add_dependencies(legacy_offset gtest_main)
add_dependencies(legacy_offset gtest)
add_sycl_to_target(
  TARGET legacy_offset
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/offset.cc
)
add_test(OffsetTests legacy_offset)
