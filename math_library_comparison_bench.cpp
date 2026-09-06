#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cassert>

// 1. GLM Includes
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // Vulkan Depth Range
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// 2. Eigen Includes
#include <Eigen/Dense>

// 3. Custom Hermetic Type0::Math
#include "Type0Math.hpp"

// FNV-1a Bitwise Checksum for raw float buffers
template<typename T>
uint64_t ComputeRawBufferHash(const T* data, size_t elementCount) {
    uint64_t hash = 14695981039346656037ULL;
    const uint8_t* bytePtr = reinterpret_cast<const uint8_t*>(data);
    const size_t totalBytes = elementCount * sizeof(T);

    for (size_t i = 0; i < totalBytes; ++i) {
        hash ^= bytePtr[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "  Math Library Exploration & Benchmark: GLM vs Eigen vs Type0::Math\n";
    std::cout << "========================================================================\n\n";

    // -------------------------------------------------------------------------
    // 1. Memory Layout & Vulkan Alignment Inspection
    // -------------------------------------------------------------------------
    std::cout << "[1] Structural Layout & Vulkan std430 Compatibility Analysis:\n";
    std::cout << "    GLM:\n";
    std::cout << "      sizeof(glm::vec3) = " << sizeof(glm::vec3) << " bytes (std430 Requires: 16 - Padding Trap!)\n";
    std::cout << "      sizeof(glm::vec4) = " << sizeof(glm::vec4) << " bytes\n";
    std::cout << "      sizeof(glm::mat4) = " << sizeof(glm::mat4) << " bytes\n";
    
    std::cout << "    Eigen:\n";
    std::cout << "      sizeof(Eigen::Vector3f) = " << sizeof(Eigen::Vector3f) << " bytes\n";
    std::cout << "      sizeof(Eigen::Vector4f) = " << sizeof(Eigen::Vector4f) << " bytes\n";
    std::cout << "      sizeof(Eigen::Matrix4f) = " << sizeof(Eigen::Matrix4f) << " bytes\n";

    std::cout << "    Type0::Math (Hermetic HWY):\n";
    std::cout << "      sizeof(Type0::Math::Vec4) = " << sizeof(Type0::Math::Vec4) << " bytes (alignof: " << alignof(Type0::Math::Vec4) << ")\n";
    std::cout << "      sizeof(Type0::Math::Mat4) = " << sizeof(Type0::Math::Mat4) << " bytes (alignof: " << alignof(Type0::Math::Mat4) << ")\n\n";

    constexpr size_t elementCount = 1'000'000;
    std::cout << "[2] Benchmark Setup: Processing " << elementCount << " 4D Transforms (Mat4 * Vec4)...\n\n";

    // Standard uniform RNG seed for deterministic input generation
    std::default_random_engine rng(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);

    // -------------------------------------------------------------------------
    // 2. GLM Transformation Benchmark
    // -------------------------------------------------------------------------
    std::cout << "[3] Evaluating GLM (MIT)..." << std::endl;
    glm::mat4 glmMat = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
    std::vector<glm::vec4> glmInput(elementCount);
    std::vector<glm::vec4> glmOutput(elementCount);

    for (size_t i = 0; i < elementCount; ++i) {
        glmInput[i] = glm::vec4(dist(rng), dist(rng), dist(rng), 1.0f);
    }

    auto startGLM = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < elementCount; ++i) {
        glmOutput[i] = glmMat * glmInput[i];
    }
    auto endGLM = std::chrono::high_resolution_clock::now();
    double glmMs = std::chrono::duration<double, std::milli>(endGLM - startGLM).count();
    uint64_t glmHash = ComputeRawBufferHash(glmOutput.data(), elementCount);

    std::cout << "    GLM Time     : " << std::fixed << std::setprecision(3) << glmMs << " ms\n";
    std::cout << "    GLM Checksum : 0x" << std::hex << glmHash << std::dec << "\n\n";

    // -------------------------------------------------------------------------
    // 3. Eigen Transformation Benchmark (Default SIMD Vectorized)
    // -------------------------------------------------------------------------
    std::cout << "[4] Evaluating Eigen (Vectorized)..." << std::endl;
    Eigen::Matrix4f eigenMat;
    std::memcpy(eigenMat.data(), glm::value_ptr(glmMat), 16 * sizeof(float));

    std::vector<Eigen::Vector4f> eigenInput(elementCount);
    std::vector<Eigen::Vector4f> eigenOutput(elementCount);

    for (size_t i = 0; i < elementCount; ++i) {
        eigenInput[i] = Eigen::Vector4f(glmInput[i].x, glmInput[i].y, glmInput[i].z, glmInput[i].w);
    }

    auto startEigen = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < elementCount; ++i) {
        eigenOutput[i] = eigenMat * eigenInput[i];
    }
    auto endEigen = std::chrono::high_resolution_clock::now();
    double eigenMs = std::chrono::duration<double, std::milli>(endEigen - startEigen).count();
    uint64_t eigenHash = ComputeRawBufferHash(eigenOutput.data(), elementCount);

    std::cout << "    Eigen Time     : " << std::fixed << std::setprecision(3) << eigenMs << " ms\n";
    std::cout << "    Eigen Checksum : 0x" << std::hex << eigenHash << std::dec << "\n\n";

    // -------------------------------------------------------------------------
    // 4. Eigen Scalar Non-Vectorized Simulation (CI Determinism mode: EIGEN_DONT_VECTORIZE)
    // -------------------------------------------------------------------------
    std::cout << "[5] Simulating Eigen CI Non-Vectorized Path (EIGEN_DONT_VECTORIZE)..." << std::endl;
    std::vector<Eigen::Vector4f> eigenScalarOutput(elementCount);

    auto startEigenScalar = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < elementCount; ++i) {
        const float* m = eigenMat.data();
        const auto& v = eigenInput[i];
        // Explicit scalar unroll without SIMD instructions
        eigenScalarOutput[i] = Eigen::Vector4f(
            m[0]*v[0] + m[4]*v[1] + m[8]*v[2]  + m[12]*v[3],
            m[1]*v[0] + m[5]*v[1] + m[9]*v[2]  + m[13]*v[3],
            m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14]*v[3],
            m[3]*v[0] + m[7]*v[1] + m[11]*v[2] + m[15]*v[3]
        );
    }
    auto endEigenScalar = std::chrono::high_resolution_clock::now();
    double eigenScalarMs = std::chrono::duration<double, std::milli>(endEigenScalar - startEigenScalar).count();
    uint64_t eigenScalarHash = ComputeRawBufferHash(eigenScalarOutput.data(), elementCount);

    std::cout << "    Eigen CI Scalar Time : " << std::fixed << std::setprecision(3) << eigenScalarMs << " ms\n";
    std::cout << "    Eigen CI Checksum    : 0x" << std::hex << eigenScalarHash << std::dec << "\n";
    std::cout << "    CI Non-Vectorized Overhead vs Vectorized Eigen: " << std::setprecision(2) << (eigenScalarMs / eigenMs) << "x slowdown\n\n";

    // -------------------------------------------------------------------------
    // 5. Custom Hermetic Type0::Math (Google Highway SIMD)
    // -------------------------------------------------------------------------
    std::cout << "[6] Evaluating Custom Hermetic Type0::Math (Google Highway)..." << std::endl;
    Type0::Math::Mat4 t0Mat;
    std::memcpy(reinterpret_cast<void*>(t0Mat.Data()), glm::value_ptr(glmMat), 16 * sizeof(float));


    std::vector<Type0::Math::Vec4> t0Input(elementCount);
    std::vector<Type0::Math::Vec4> t0Output(elementCount);

    for (size_t i = 0; i < elementCount; ++i) {
        t0Input[i] = {glmInput[i].x, glmInput[i].y, glmInput[i].z, glmInput[i].w};
    }

    auto startT0 = std::chrono::high_resolution_clock::now();
    Type0::Math::SIMD::BatchMatVecTransform(t0Mat, t0Input.data(), t0Output.data(), elementCount);
    auto endT0 = std::chrono::high_resolution_clock::now();
    double t0Ms = std::chrono::duration<double, std::milli>(endT0 - startT0).count();
    uint64_t t0Hash = ComputeRawBufferHash(t0Output.data(), elementCount);

    std::cout << "    Type0::Math Time     : " << std::fixed << std::setprecision(3) << t0Ms << " ms\n";
    std::cout << "    Type0::Math Checksum : 0x" << std::hex << t0Hash << std::dec << "\n";
    std::cout << "    Speedup vs GLM       : " << std::setprecision(2) << (glmMs / t0Ms) << "x\n";
    std::cout << "    Speedup vs Eigen CI  : " << std::setprecision(2) << (eigenScalarMs / t0Ms) << "x\n\n";

    // -------------------------------------------------------------------------
    // Summary Matrix Output
    // -------------------------------------------------------------------------
    std::cout << "========================================================================\n";
    std::cout << "  EVALUATION SUMMARY & COMPARISON MATRIX\n";
    std::cout << "========================================================================\n";
    std::cout << " Framework              | Execution Time | Checksum           | Vulkan std430 Alignment\n";
    std::cout << " -----------------------+----------------+--------------------+-----------------------\n";
    std::cout << " GLM (MIT)              | " << std::setw(8) << glmMs << " ms | 0x" << std::hex << glmHash << std::dec << " | Requires padding wrapper\n";
    std::cout << " Eigen (Vectorized)     | " << std::setw(8) << eigenMs << " ms | 0x" << std::hex << eigenHash << std::dec << " | Column-Major, non-std430\n";
    std::cout << " Eigen (CI Scalar)      | " << std::setw(8) << eigenScalarMs << " ms | 0x" << std::hex << eigenScalarHash << std::dec << " | Non-Vectorized Parity\n";
    std::cout << " Type0::Math (HWY SIMD) | " << std::setw(8) << t0Ms << " ms | 0x" << std::hex << t0Hash << std::dec << " | Native alignas(16) std430\n";
    std::cout << "========================================================================\n";

    return 0;
}
