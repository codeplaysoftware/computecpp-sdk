set(SOURCE_NAME "using_functors")

include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

add_executable(${SOURCE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp)

add_sycl_to_target(${SOURCE_NAME}  ${CMAKE_CURRENT_BINARY_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp)

add_test(NAME ${SOURCE_NAME} COMMAND ${SOURCE_NAME})
