project(MandelbrotApp)

set(SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/mandel.cpp")

ci_make_app(
  APP_NAME "Mandelbrot"
  SOURCES "${SOURCES}"
  LIBRARIES "${OpenCL_LIBRARIES}"
  CINDER_PATH "${CINDER_PATH}")

add_sycl_to_target(
    TARGET "Mandelbrot"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/mandel.cpp")
