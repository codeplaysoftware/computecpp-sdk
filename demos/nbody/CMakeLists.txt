add_executable(NBody WIN32 main.cpp
                           sim.cpp)

link_demo_dependencies(TARGET NBody)

# Copy the assets folder to the binary folder so the program can load it.
add_custom_command(TARGET NBody POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets ${CMAKE_CURRENT_BINARY_DIR}/assets)

add_sycl_to_target(TARGET  NBody
                   SOURCES sim.cpp)
