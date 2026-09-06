cmake_minimum_required(VERSION 3.14)
project(HermeticTesting)

set(CMAKE_CXX_STANDARD 20)

set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(highway)

add_executable(VerifyHighwayHermeticMath VerifyHighwayHermeticMath.cpp)
target_link_libraries(VerifyHighwayHermeticMath PRIVATE hwy)
target_include_directories(VerifyHighwayHermeticMath PRIVATE ${highway_SOURCE_DIR})

add_executable(BitwiseAndSpeedComparison BitwiseAndSpeedComparison.cpp)
target_link_libraries(BitwiseAndSpeedComparison PRIVATE hwy)
target_include_directories(BitwiseAndSpeedComparison PRIVATE ${highway_SOURCE_DIR})

add_executable(CompareAllFunctionsAndTestingGuide CompareAllFunctionsAndTestingGuide.cpp)
target_link_libraries(CompareAllFunctionsAndTestingGuide PRIVATE hwy)
target_include_directories(CompareAllFunctionsAndTestingGuide PRIVATE ${highway_SOURCE_DIR})

add_executable(CrossHardwareMatrixTest CrossHardwareMatrixTest.cpp)
target_link_libraries(CrossHardwareMatrixTest PRIVATE hwy)
target_include_directories(CrossHardwareMatrixTest PRIVATE ${highway_SOURCE_DIR})

add_executable(HermeticAudioTest HermeticAudioTest.cpp)
target_include_directories(HermeticAudioTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(DirectPerformanceComparison DirectPerformanceComparison.cpp)
target_link_libraries(DirectPerformanceComparison PRIVATE hwy)
target_include_directories(DirectPerformanceComparison PRIVATE ${highway_SOURCE_DIR})
if(MSVC)
    target_compile_options(DirectPerformanceComparison PRIVATE /O2 /arch:AVX2 /fp:precise)
else()
    target_compile_options(DirectPerformanceComparison PRIVATE -O3 -mavx2 -mfma)
endif()

find_path(VULKAN_HEADERS_INCLUDE_DIR NAMES vulkan/vulkan.h HINTS "C:/vcpkg/packages/vulkan-headers_x64-windows/include")
find_library(VULKAN_LIB NAMES vulkan-1 HINTS "C:/vcpkg/packages/vulkan-loader_x64-windows/lib")

add_executable(RealVulkanComputeBenchmark RealVulkanComputeBenchmark.cpp)
target_include_directories(RealVulkanComputeBenchmark PRIVATE ${highway_SOURCE_DIR} ${VULKAN_HEADERS_INCLUDE_DIR})
target_link_libraries(RealVulkanComputeBenchmark PRIVATE hwy ${VULKAN_LIB})
if(MSVC)
    target_compile_options(RealVulkanComputeBenchmark PRIVATE /O2 /arch:AVX2 /fp:precise)
endif()

add_executable(VulkanZeroCopyPipelineBenchmark VulkanZeroCopyPipelineBenchmark.cpp)
target_include_directories(VulkanZeroCopyPipelineBenchmark PRIVATE ${highway_SOURCE_DIR} ${VULKAN_HEADERS_INCLUDE_DIR})
target_link_libraries(VulkanZeroCopyPipelineBenchmark PRIVATE hwy ${VULKAN_LIB})
if(MSVC)
    target_compile_options(VulkanZeroCopyPipelineBenchmark PRIVATE /O2 /arch:AVX2 /fp:precise)
endif()




