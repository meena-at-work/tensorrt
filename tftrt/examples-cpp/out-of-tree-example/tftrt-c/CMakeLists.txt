cmake_minimum_required(VERSION 3.10)

project(tftrt_c_example)

add_executable(tftrt_c_example main.c)

target_include_directories(tftrt_c_example PUBLIC
                           "/usr/local/lib/python3.8/dist-packages/tensorflow/include")

target_link_libraries(tftrt_c_example PUBLIC "/usr/local/lib/python3.8/dist-packages/tensorflow/libtensorflow_framework.so")
target_link_libraries(tftrt_c_example PUBLIC "/usr/local/lib/tensorflow/libtensorflow_cc.so")

