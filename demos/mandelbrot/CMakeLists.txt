project(MandelbrotApp)

set(SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/mandel.cpp")

ci_make_app(
  APP_NAME "Mandelbrot"
  SOURCES "${SOURCES}"
  LIBRARIES "${OpenCL_LIBRARIES}"
  LIBRARIES "${COMPUTECPP_RUNTIME_LIBRARY}"
  CINDER_PATH "${CINDER_PATH}")

__build_spir("Mandelbrot" "${CMAKE_CURRENT_SOURCE_DIR}/mandel.cpp" "${CMAKE_CURRENT_BINARY_DIR}" 1)
