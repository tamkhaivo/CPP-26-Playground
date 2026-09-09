cmake_minimum_required(VERSION 3.14)
project(SimdBenchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)

# Fetch Google Highway
include(FetchContent)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(highway)

# Find Threads support
find_package(Threads REQUIRED)

add_executable(simd_benchmark main.cpp)
target_link_libraries(simd_benchmark PRIVATE hwy Threads::Threads)
if(MSVC)
    target_compile_options(simd_benchmark PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(simd_benchmark PRIVATE -O3 -mavx2)
endif()
