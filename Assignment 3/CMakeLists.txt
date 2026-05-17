cmake_minimum_required(VERSION 3.10)
project(Assignment3)

set(CMAKE_CXX_STANDARD 20)

# Create the plugins directory inside the build output
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/plugins)

# Compile PluginA into the plugins/ directory
add_library(PluginA SHARED PluginA.cpp)
set_target_properties(PluginA PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/plugins)

# Compile PluginB into the plugins/ directory
add_library(PluginB SHARED PluginB.cpp)
set_target_properties(PluginB PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/plugins)

# Build the main executable
add_executable(Assignment3 main.cpp)
