cmake_minimum_required(VERSION 3.20)
project(StratifiedPhysicsExploration CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

include(FetchContent)

# Fetch Google Highway
FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG        1.1.0
)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)

# Fetch GLM
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# Fetch Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(eigen)


# Fetch Jolt Physics
FetchContent_Declare(
    JoltPhysics
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG        v5.2.0
    SOURCE_SUBDIR  Build
)
set(CROSS_PLATFORM_DETERMINISTIC ON CACHE BOOL "" FORCE)
set(INTERMEDIATE_CLOTH OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(PORTABLE_COMPILER_FLAGS ON CACHE BOOL "" FORCE)
set(USE_STATIC_C_RUNTIME ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(JoltPhysics)

find_package(Threads REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(stratified_physics_exploration stratified_physics_exploration.cpp)
target_include_directories(stratified_physics_exploration PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(stratified_physics_exploration PRIVATE Jolt hwy hwy_contrib Threads::Threads tbb ${Vulkan_LIBRARIES})

add_executable(asset_cooker AssetCooker.cpp)
target_link_libraries(asset_cooker PRIVATE hwy hwy_contrib)

add_executable(hermetic_math_benchmark hermetic_math_benchmark.cpp)
target_link_libraries(hermetic_math_benchmark PRIVATE hwy hwy_contrib Threads::Threads tbb)

add_executable(math_library_comparison_bench math_library_comparison_bench.cpp)
target_link_libraries(math_library_comparison_bench PRIVATE glm::glm Eigen3::Eigen hwy hwy_contrib Threads::Threads)

add_executable(test_aligned_soa_simd test_aligned_soa_simd.cpp)
target_link_libraries(test_aligned_soa_simd PRIVATE hwy hwy_contrib Threads::Threads tbb)

if (MSVC)
    target_compile_options(stratified_physics_exploration PRIVATE /O2 /arch:AVX2 /fp:precise /MT)
    target_compile_options(asset_cooker PRIVATE /O2 /arch:AVX2 /MT)
    target_compile_options(hermetic_math_benchmark PRIVATE /O2 /arch:AVX2 /MT)
    target_compile_options(math_library_comparison_bench PRIVATE /O2 /arch:AVX2 /MT)
    target_compile_options(test_aligned_soa_simd PRIVATE /O2 /arch:AVX2 /MT)
else()
    target_compile_options(stratified_physics_exploration PRIVATE -O3 -mavx2 -mfma)
    target_compile_options(asset_cooker PRIVATE -O3 -mavx2)
    target_compile_options(hermetic_math_benchmark PRIVATE -O3 -mavx2)
    target_compile_options(math_library_comparison_bench PRIVATE -O3 -mavx2)
    target_compile_options(test_aligned_soa_simd PRIVATE -O3 -mavx2)
endif()





