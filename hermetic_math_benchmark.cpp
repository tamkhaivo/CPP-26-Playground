#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <execution>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cassert>

#include "Type0Math.hpp"

using namespace Type0::Math;

void RunScalarBaseline(const std::vector<Vec4>& inA,
                       const std::vector<Vec4>& inB,
                       float scale,
                       std::vector<Vec4>& outVec) {
    const size_t n = inA.size();
    for (size_t i = 0; i < n; ++i) {
        outVec[i].x = inA[i].x * scale + inB[i].x;
        outVec[i].y = inA[i].y * scale + inB[i].y;
        outVec[i].z = inA[i].z * scale + inB[i].z;
        outVec[i].w = inA[i].w * scale + inB[i].w;
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  Type0 Engine: Hermetic SIMD Math & Vulkan 1.4 Benchmark\n";
    std::cout << "========================================================\n\n";

    // 1. Memory Layout Assertions for Vulkan std430 compatibility
    std::cout << "[1] Verifying Vulkan 1.4 Memory Layout & Byte Alignment...\n";
    std::cout << "    sizeof(Vec4)  = " << sizeof(Vec4) << " bytes (Required: 16)\n";
    std::cout << "    alignof(Vec4) = " << alignof(Vec4) << " bytes (Required: 16)\n";
    std::cout << "    sizeof(Mat4)  = " << sizeof(Mat4) << " bytes (Required: 64)\n";
    std::cout << "    sizeof(AABB)  = " << sizeof(AABB) << " bytes (Required: 32)\n";

    assert(sizeof(Vec4) == 16);
    assert(alignof(Vec4) == 16);
    assert(sizeof(Mat4) == 64);
    assert(sizeof(AABB) == 32);
    std::cout << "    -> Layout Assertions PASSED successfully.\n\n";

    // 2. Data Preparation
    constexpr size_t elementCount = 1'000'000;
    std::cout << "[2] Initializing dataset with " << elementCount << " Vec4 elements...\n";

    std::vector<Vec4> inA(elementCount);
    std::vector<Vec4> inB(elementCount);
    std::vector<Vec4> outScalar(elementCount);
    std::vector<Vec4> outSIMD(elementCount);
    std::vector<Vec4> outParSIMD(elementCount);

    std::default_random_engine rng(1337); // Fixed seed for bit determinism
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (size_t i = 0; i < elementCount; ++i) {
        inA[i] = {dist(rng), dist(rng), dist(rng), dist(rng)};
        inB[i] = {dist(rng), dist(rng), dist(rng), dist(rng)};
    }
    const float scale = 2.5f;

    // 3. Scalar Baseline Performance Benchmark
    std::cout << "[3] Running Scalar Baseline Loop...\n";
    auto startScalar = std::chrono::high_resolution_clock::now();
    RunScalarBaseline(inA, inB, scale, outScalar);
    auto endScalar = std::chrono::high_resolution_clock::now();
    double scalarMs = std::chrono::duration<double, std::milli>(endScalar - startScalar).count();
    uint64_t hashScalar = ComputeDeterminismChecksum(outScalar.data(), elementCount);
    std::cout << "    Scalar Time: " << std::fixed << std::setprecision(3) << scalarMs << " ms\n";
    std::cout << "    Scalar Checksum (FNV-1a): 0x" << std::hex << hashScalar << std::dec << "\n\n";

    // 4. Single-Threaded Highway SIMD Benchmark
    std::cout << "[4] Running Single-Threaded Google Highway SIMD Loop...\n";
    auto startSIMD = std::chrono::high_resolution_clock::now();
    SIMD::BatchMulAdd(inA.data(), inB.data(), scale, outSIMD.data(), elementCount);
    auto endSIMD = std::chrono::high_resolution_clock::now();
    double simdMs = std::chrono::duration<double, std::milli>(endSIMD - startSIMD).count();
    uint64_t hashSIMD = ComputeDeterminismChecksum(outSIMD.data(), elementCount);
    std::cout << "    Highway SIMD Time: " << std::fixed << std::setprecision(3) << simdMs << " ms\n";
    std::cout << "    Highway SIMD Checksum: 0x" << std::hex << hashSIMD << std::dec << "\n";
    std::cout << "    Single-Thread Speedup vs Scalar: " << std::setprecision(2) << (scalarMs / simdMs) << "x\n\n";

    // 5. Parallel Multi-Threaded Highway SIMD Benchmark (std::execution::par)
    std::cout << "[5] Running Parallel Multi-Threaded Highway SIMD Loop (std::execution::par)...\n";
    const size_t chunkSize = 16384;
    const size_t totalChunks = (elementCount + chunkSize - 1) / chunkSize;
    std::vector<size_t> chunkIndices(totalChunks);
    std::iota(chunkIndices.begin(), chunkIndices.end(), 0);

    auto startParSIMD = std::chrono::high_resolution_clock::now();
    std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
        size_t start = chunkIdx * chunkSize;
        size_t count = std::min(chunkSize, elementCount - start);
        SIMD::BatchMulAdd(inA.data() + start, inB.data() + start, scale, outParSIMD.data() + start, count);
    });
    auto endParSIMD = std::chrono::high_resolution_clock::now();
    double parSimdMs = std::chrono::duration<double, std::milli>(endParSIMD - startParSIMD).count();
    uint64_t hashParSIMD = ComputeDeterminismChecksum(outParSIMD.data(), elementCount);

    std::cout << "    Parallel SIMD Time: " << std::fixed << std::setprecision(3) << parSimdMs << " ms\n";
    std::cout << "    Parallel SIMD Checksum: 0x" << std::hex << hashParSIMD << std::dec << "\n";
    std::cout << "    Parallel Speedup vs Scalar: " << std::setprecision(2) << (scalarMs / parSimdMs) << "x\n\n";

    // 6. Cross-Execution Bit Determinism Verification
    std::cout << "[6] Verifying Cross-Execution Bit Determinism...\n";
    std::cout << "    Scalar Checksum   : 0x" << std::hex << hashScalar << std::dec << "\n";
    std::cout << "    SIMD Checksum     : 0x" << std::hex << hashSIMD << std::dec << "\n";
    std::cout << "    Par-SIMD Checksum : 0x" << std::hex << hashParSIMD << std::dec << "\n";

    if (hashSIMD == hashParSIMD) {
        std::cout << "    -> SUCCESS: SIMD and Parallel SIMD hashes match 100% bit-for-bit!\n";
    } else {
        std::cout << "    -> WARNING: Bit mismatch between sequential and parallel SIMD execution.\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << "  Benchmark Execution Complete\n";
    std::cout << "========================================================\n";

    return 0;
}
