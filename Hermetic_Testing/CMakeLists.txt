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
if(MSVC)
    target_link_libraries(DirectPerformanceComparison PRIVATE hwy)
    target_compile_options(DirectPerformanceComparison PRIVATE /O2 /arch:AVX2 /fp:precise)
else()
    target_link_libraries(DirectPerformanceComparison PRIVATE hwy tbb)
    target_compile_options(DirectPerformanceComparison PRIVATE -O3 -mavx2 -mfma)
endif()
target_include_directories(DirectPerformanceComparison PRIVATE ${highway_SOURCE_DIR})

find_package(Vulkan REQUIRED)

add_executable(RealVulkanComputeBenchmark RealVulkanComputeBenchmark.cpp)
target_include_directories(RealVulkanComputeBenchmark PRIVATE ${highway_SOURCE_DIR} ${Vulkan_INCLUDE_DIRS})
if(MSVC)
    target_link_libraries(RealVulkanComputeBenchmark PRIVATE hwy ${Vulkan_LIBRARIES})
    target_compile_options(RealVulkanComputeBenchmark PRIVATE /O2 /arch:AVX2 /fp:precise)
else()
    target_link_libraries(RealVulkanComputeBenchmark PRIVATE hwy ${Vulkan_LIBRARIES} tbb)
    target_compile_options(RealVulkanComputeBenchmark PRIVATE -O3 -mavx2)
endif()

add_executable(VulkanZeroCopyPipelineBenchmark VulkanZeroCopyPipelineBenchmark.cpp)
target_include_directories(VulkanZeroCopyPipelineBenchmark PRIVATE ${highway_SOURCE_DIR} ${Vulkan_INCLUDE_DIRS})
if(MSVC)
    target_link_libraries(VulkanZeroCopyPipelineBenchmark PRIVATE hwy ${Vulkan_LIBRARIES})
    target_compile_options(VulkanZeroCopyPipelineBenchmark PRIVATE /O2 /arch:AVX2 /fp:precise)
else()
    target_link_libraries(VulkanZeroCopyPipelineBenchmark PRIVATE hwy ${Vulkan_LIBRARIES} tbb)
    target_compile_options(VulkanZeroCopyPipelineBenchmark PRIVATE -O3 -mavx2)
endif()

add_executable(SIMDDeterminismTest SIMDDeterminismTest.cpp)
target_link_libraries(SIMDDeterminismTest PRIVATE hwy)
target_include_directories(SIMDDeterminismTest PRIVATE ${highway_SOURCE_DIR})
if(NOT MSVC)
    target_compile_options(SIMDDeterminismTest PRIVATE -O3 -mavx2)
endif()

add_executable(BattleTestFloatDeterminism BattleTestFloatDeterminism.cpp)
target_link_libraries(BattleTestFloatDeterminism PRIVATE hwy)
target_include_directories(BattleTestFloatDeterminism PRIVATE ${highway_SOURCE_DIR})
if(NOT MSVC)
    target_compile_options(BattleTestFloatDeterminism PRIVATE -O3 -mavx2)
endif()

add_executable(BattleTestDeepEngineHoles BattleTestDeepEngineHoles.cpp)
target_link_libraries(BattleTestDeepEngineHoles PRIVATE hwy)
target_include_directories(BattleTestDeepEngineHoles PRIVATE ${highway_SOURCE_DIR})
if(NOT MSVC)
    target_compile_options(BattleTestDeepEngineHoles PRIVATE -O3 -mavx2)
endif()
