cmake_minimum_required(VERSION 3.20)
project(BenchmarkCPP CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Fetch Google Highway
FetchContent_Declare(
    hwy
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG        1.1.0
)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(hwy)

# Fetch enkiTS
FetchContent_Declare(
    enkiTS
    GIT_REPOSITORY https://github.com/dougbinks/enkiTS.git
    GIT_TAG        master
)
set(ENKITS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ENKITS_BUILD_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(enkiTS)

add_executable(benchmark_cpp main.cpp)
target_link_libraries(benchmark_cpp PRIVATE hwy enkiTS)
target_compile_options(benchmark_cpp PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3>)
