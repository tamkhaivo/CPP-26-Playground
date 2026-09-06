cmake_minimum_required(VERSION 3.20)
project(AllPhysicsEnginesPerfBenchmark CXX)

# Fix CMP0000 / CMP0048 policy deprecations for legacy dependencies like Bullet 3
if (POLICY CMP0000)
    cmake_policy(SET CMP0000 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

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

# Fetch Jolt Physics
FetchContent_Declare(
    JoltPhysics
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG        v5.2.0
    SOURCE_SUBDIR  Build
)
set(CROSS_PLATFORM_DETERMINISTIC ON CACHE BOOL "" FORCE)
set(INTERMEDIATE_CLOTH ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(PORTABLE_COMPILER_FLAGS ON CACHE BOOL "" FORCE)
set(USE_STATIC_C_RUNTIME ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(JoltPhysics)

add_executable(all_engines_perf_benchmark all_perf/main_all_perf.cpp)
target_link_libraries(all_engines_perf_benchmark PRIVATE Jolt hwy enkiTS)

if (MSVC)
    target_compile_options(all_engines_perf_benchmark PRIVATE /O2 /fp:fast /arch:AVX2 /MT)
else()
    target_compile_options(all_engines_perf_benchmark PRIVATE -O3 -ffast-math -mavx2)
endif()
