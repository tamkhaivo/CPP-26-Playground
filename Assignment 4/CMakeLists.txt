# Enable policy compatibility for older subprojects (like enkiTS) on newer CMake versions
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

cmake_minimum_required(VERSION 3.14)
project(Assignment4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable Windows min/max macro expansion to prevent collisions with std::min/max
add_compile_definitions(NOMINMAX)

# Configure FetchContent options to disable testing/examples for dependency libraries
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)

set(TBB_TEST OFF CACHE BOOL "" FORCE)
set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TBB_STRICT OFF CACHE BOOL "" FORCE)

set(ENKITS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ENKITS_BUILD_SHARED OFF CACHE BOOL "" FORCE)

set(TF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(MARL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MARL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(STDEXEC_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(STDEXEC_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

include(FetchContent)

# 1. Fetch Google Highway
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        master
)

# 2. Fetch oneTBB
FetchContent_Declare(
  tbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG        v2021.12.0
)

# 3. Fetch enkiTS
FetchContent_Declare(
  enkits
  GIT_REPOSITORY https://github.com/dougbinks/enkiTS.git
  GIT_TAG        v1.11
)

# 4. Fetch Taskflow
FetchContent_Declare(
  taskflow
  GIT_REPOSITORY https://github.com/taskflow/taskflow.git
  GIT_TAG        v3.7.0
)

# 5. Fetch Marl (main branch is used since master branch was removed)
FetchContent_Declare(
  marl
  GIT_REPOSITORY https://github.com/google/marl.git
  GIT_TAG        main
)

# 6. Fetch stdexec
FetchContent_Declare(
  stdexec
  GIT_REPOSITORY https://github.com/NVIDIA/stdexec.git
  GIT_TAG        main
)

# Make all dependencies available
FetchContent_MakeAvailable(highway tbb enkits taskflow marl stdexec)

# Find Threads support
find_package(Threads REQUIRED)

# Compile dynamic library plugin
add_library(ParticlePhysics SHARED ParticlePhysics.cpp)
target_include_directories(ParticlePhysics PRIVATE "${enkits_SOURCE_DIR}/src")
target_link_libraries(ParticlePhysics PRIVATE 
  hwy 
  TBB::tbb 
  enkiTS 
  Taskflow 
  marl 
  stdexec 
  Threads::Threads
)

# Compile main benchmark harness executable
add_executable(Assignment4 main.cpp)
target_include_directories(Assignment4 PRIVATE "${enkits_SOURCE_DIR}/src")
target_link_libraries(Assignment4 PRIVATE 
  TBB::tbb 
  enkiTS 
  Taskflow 
  marl 
  stdexec 
  Threads::Threads
)

# Compile Job System Benchmark target
add_executable(job_system_benchmark job_system_benchmark.cpp)
target_include_directories(job_system_benchmark PRIVATE "${enkits_SOURCE_DIR}/src")
target_link_libraries(job_system_benchmark PRIVATE
  hwy
  enkiTS
  Taskflow
  Threads::Threads
)

# Compile Hermetic Job System Verifier target
add_executable(job_system_hermetic_verifier job_system_hermetic_verifier.cpp)
target_include_directories(job_system_hermetic_verifier PRIVATE "${enkits_SOURCE_DIR}/src")
target_link_libraries(job_system_hermetic_verifier PRIVATE
  hwy
  enkiTS
  Taskflow
  Threads::Threads
)


