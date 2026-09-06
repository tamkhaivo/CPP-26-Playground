set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

FetchContent_Declare(
    JoltPhysicsFast
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG        v5.2.0
    SOURCE_SUBDIR  Build
)
set(CROSS_PLATFORM_DETERMINISTIC OFF CACHE BOOL "" FORCE)
set(INTERMEDIATE_CLOTH ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(PORTABLE_COMPILER_FLAGS OFF CACHE BOOL "" FORCE)
set(USE_STATIC_C_RUNTIME ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(JoltPhysicsFast)

add_executable(jolt_fast main_fast.cpp)
target_link_libraries(jolt_fast PRIVATE Jolt)

if (MSVC)
    target_compile_options(jolt_fast PRIVATE /O2 /fp:fast /MT)
else()
    target_compile_options(jolt_fast PRIVATE -O3 -ffast-math)
endif()
