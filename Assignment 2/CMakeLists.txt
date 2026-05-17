cmake_minimum_required(VERSION 3.14)
project(Assignment2)

set(CMAKE_CXX_STANDARD 20)

set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)

# Fetch Google Highway directly from GitHub so you don't have to install it manually!
include(FetchContent)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(highway)

add_executable(Assignment2 main.cpp)
target_link_libraries(Assignment2 PRIVATE hwy)
