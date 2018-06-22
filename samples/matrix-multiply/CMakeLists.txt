set(SOURCE_NAME "matrix-multiply")
add_executable(
  ${SOURCE_NAME}
  ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
)
if(NOT MSVC)
  target_compile_options(
    ${SOURCE_NAME}
    PUBLIC -Wno-unknown-pragmas
  )

  if(COMPUTECPP_SDK_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(${SOURCE_NAME} PRIVATE OpenMP::OpenMP_CXX)
    else()
        # Workaround for CMake <3.9 - FindOpenMP does not provide the
        # linker flags and no imported target
        target_compile_options(
          ${SOURCE_NAME}
          PRIVATE ${OpenMP_CXX_FLAGS}
        )
        target_link_libraries(
          ${SOURCE_NAME}
          PRIVATE ${OpenMP_CXX_FLAGS}
        )
    endif()
  endif()
endif()
add_sycl_to_target(
  TARGET ${SOURCE_NAME}
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp
)
add_test(
  NAME ${SOURCE_NAME}_omp
  COMMAND ${SOURCE_NAME} 64 omp
)
add_test(
  NAME ${SOURCE_NAME}_sycl
  COMMAND ${SOURCE_NAME} 64 sycl
)
install(
  TARGETS ${SOURCE_NAME}
  RUNTIME DESTINATION bin
)
