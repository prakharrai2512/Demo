# CMakeList.txt : CMake project for Demo, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Demo")

set(CMAKE_PREFIX_PATH "C:\\libtorch-win-shared-with-deps-1.11.0+cu113\\libtorch")
link_directories("C:\\libtorch-win-shared-with-deps-1.11.0+cu113\\libtorch\\include" "C:\\libtorch-win-shared-with-deps-1.11.0+cu113\\libtorch\\include\\torch\\csrc\\api\\include")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# Add source to this project's executable.
add_executable (Demo "Demo.cpp" "Demo.h" "DemoGPU.cpp")

target_link_libraries(Demo "${TORCH_LIBRARIES}")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Demo PROPERTY CXX_STANDARD 17)
endif()

# the DLLs need to be copied to avoid memory errors.

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET Demo
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:Demo>)
endif (MSVC)