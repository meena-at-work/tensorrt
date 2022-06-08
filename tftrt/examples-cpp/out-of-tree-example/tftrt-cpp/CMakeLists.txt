cmake_minimum_required(VERSION 3.10)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(tftrt_cpp_example)

add_executable(tftrt_cpp_example tftrt_cpp_example.cpp tftrt_nn_runner.cpp)

target_include_directories(tftrt_cpp_example PUBLIC
                           "/usr/local/lib/python3.8/dist-packages/tensorflow/include")

target_link_libraries(tftrt_cpp_example PUBLIC "/usr/local/lib/tensorflow/libtensorflow_cc.so")
target_link_libraries(tftrt_cpp_example PUBLIC "/usr/local/lib/python3.8/dist-packages/tensorflow/libtensorflow_framework.so")

