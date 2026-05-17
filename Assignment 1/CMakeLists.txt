cmake_minimum_required(VERSION 3.10)
project(Assignment1)

set(CMAKE_CXX_STANDARD 20)

# Build the dynamic library (the mock Nvidia plugin)
add_library(DynamicLoading_NvidiaStreamline SHARED DynamicLoading_NvidiaStreamline.cpp)

# Build the main engine executable
add_executable(Assignment1 main.cpp)
