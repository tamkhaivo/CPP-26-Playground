set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_compile_definitions(JPH_CROSS_PLATFORM_DETERMINISTIC)

if (MSVC)
    add_compile_options(/fp:strict /MT)
    string(REGEX REPLACE "/fp:fast" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "/fp:fast" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REGEX REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
else()
    add_compile_options(-ffp-contract=off)
    string(REGEX REPLACE "-ffast-math" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "-ffast-math" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

FetchContent_Declare(
    JoltPhysicsDet
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG        v5.2.0
    SOURCE_SUBDIR  Build
)
set(CROSS_PLATFORM_DETERMINISTIC ON CACHE BOOL "" FORCE)
set(INTERMEDIATE_CLOTH ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(PORTABLE_COMPILER_FLAGS ON CACHE BOOL "" FORCE)
set(USE_STATIC_C_RUNTIME ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(JoltPhysicsDet)

add_executable(jolt_deterministic main_det.cpp)
target_link_libraries(jolt_deterministic PRIVATE Jolt hwy)

if (MSVC)
    target_compile_options(jolt_deterministic PRIVATE /O2 /fp:strict /MT)
else()
    target_compile_options(jolt_deterministic PRIVATE -O3 -ffp-contract=off)
endif()
