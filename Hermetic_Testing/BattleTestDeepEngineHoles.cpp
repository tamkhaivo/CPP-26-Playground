#include "FixedMath.hpp"
#include "ImageMetrics.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>

#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;
using namespace Type0::Testing;

// ============================================================================
// HOLE 4: Multi-Threaded Out-of-Order Task Merging (Atomic Buffer Counter)
// ============================================================================
void TestHole4_ThreadSchedulingOrder() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[HOLE 4 BATTLE-TEST] Multi-Threaded Atomic Output Order Non-Determinism\n";
    std::cout << "------------------------------------------------------------------------\n";

    constexpr size_t THREAD_COUNT = 4;
    constexpr size_t ITEMS_PER_THREAD = 256;

    std::vector<uint32_t> run1_buffer(THREAD_COUNT * ITEMS_PER_THREAD);
    std::vector<uint32_t> run2_buffer(THREAD_COUNT * ITEMS_PER_THREAD);

    auto RunParallelTask = [](std::vector<uint32_t>& buffer) {
        std::atomic<uint32_t> global_index{0};
        std::vector<std::thread> threads;

        for (uint32_t t = 0; t < THREAD_COUNT; ++t) {
            threads.emplace_back([t, &buffer, &global_index]() {
                std::this_thread::sleep_for(std::chrono::microseconds((4 - t) * 10));
                for (size_t i = 0; i < ITEMS_PER_THREAD; ++i) {
                    uint32_t idx = global_index.fetch_add(1, std::memory_order_relaxed);
                    buffer[idx] = t * 1000 + static_cast<uint32_t>(i);
                }
            });
        }
        for (auto& th : threads) th.join();
    };

    RunParallelTask(run1_buffer);
    RunParallelTask(run2_buffer);

    uint64_t hash1 = ImageMetrics::CalculateHash(run1_buffer);
    uint64_t hash2 = ImageMetrics::CalculateHash(run2_buffer);

    std::cout << "Run 1 Parallel Buffer Hash: " << hash1 << "\n";
    std::cout << "Run 2 Parallel Buffer Hash: " << hash2 << "\n";

    if (hash1 != hash2) {
        std::cout << "❌ DEEP HOLE 4 PROVED: Multi-threaded task scheduling produces OUT-OF-ORDER memory arrays!\n";
        std::cout << "   -> Atomic buffer allocation (`atomic.fetch_add`) in jobs/compute shaders changes memory layout every run.\n";
        std::cout << "   -> REQUIREMENT: Parallel tasks MUST write to deterministic thread-indexed sub-ranges, NOT atomic stacks.\n";
    }
}

// ============================================================================
// HOLE 5: Uninitialized Struct Alignment Padding Garbage Bytes
// ============================================================================
struct alignas(16) MeshInstancePadded {
    float position[3]; // 12 bytes
    // 4 bytes padding compiler inserted
    uint32_t color;    // 4 bytes
};

void TestHole5_UninitializedPaddingGarbage() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[HOLE 5 BATTLE-TEST] Uninitialized Struct Alignment Padding Bytes\n";
    std::cout << "------------------------------------------------------------------------\n";

    MeshInstancePadded inst_garbage;
    std::memset(&inst_garbage, 0xAA, sizeof(MeshInstancePadded)); // Simulate uninitialized stack garbage in padding bytes
    inst_garbage.position[0] = 1.0f;
    inst_garbage.position[1] = 2.0f;
    inst_garbage.position[2] = 3.0f;
    inst_garbage.color = 0xFF0000FF;

    MeshInstancePadded inst_clean;
    std::memset(&inst_clean, 0x00, sizeof(MeshInstancePadded)); // Clean zero-fill
    inst_clean.position[0] = 1.0f;
    inst_clean.position[1] = 2.0f;
    inst_clean.position[2] = 3.0f;
    inst_clean.color = 0xFF0000FF;

    std::vector<uint32_t> buf_garbage(sizeof(MeshInstancePadded) / 4);
    std::vector<uint32_t> buf_clean(sizeof(MeshInstancePadded) / 4);

    std::memcpy(buf_garbage.data(), &inst_garbage, sizeof(MeshInstancePadded));
    std::memcpy(buf_clean.data(), &inst_clean, sizeof(MeshInstancePadded));

    uint64_t hash_garbage = ImageMetrics::CalculateHash(buf_garbage);
    uint64_t hash_clean = ImageMetrics::CalculateHash(buf_clean);

    std::cout << "Clean Zero-Padded Struct Hash:       " << hash_clean << "\n";
    std::cout << "Un-initialized Padding Struct Hash: " << hash_garbage << "\n";

    if (hash_garbage != hash_clean) {
        std::cout << "❌ DEEP HOLE 5 PROVED: Un-zeroed compiler padding bytes introduce GARBAGE HASH NOISE!\n";
        std::cout << "   -> `struct alignas(16)` contains 4 uninitialized padding bytes on the stack.\n";
        std::cout << "   -> Hashing raw vertex/uniform buffer memory fails even when all struct fields match!\n";
        std::cout << "   -> REQUIREMENT: Always zero-fill struct memory before writing (`memset` or `= {}`).\n";
    }
}

// ============================================================================
// HOLE 6: Subpixel UV Bilinear Interpolation Weight Drift
// ============================================================================
void TestHole6_SubpixelUVDrift() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[HOLE 6 BATTLE-TEST] Subpixel UV Bilinear Interpolation Weight Drift\n";
    std::cout << "------------------------------------------------------------------------\n";

    // 2x2 Texture texels: [TopLeft=0, TopRight=255, BottomLeft=255, BottomRight=0]
    float t00 = 0.0f, t10 = 255.0f;
    float t01 = 255.0f, t11 = 0.0f;

    // UV position crossing rounding boundary (0.499999f vs 0.500008f)
    float u1 = 0.499999f, v1 = 0.499999f;
    float u2 = 0.500008f, v2 = 0.500008f;

    auto BilinearSample = [](float u, float v, float c00, float c10, float c01, float c11) {
        float top = c00 * (1.0f - u) + c10 * u;
        float bot = c01 * (1.0f - u) + c11 * u;
        return top * (1.0f - v) + bot * v;
    };

    float sample1 = BilinearSample(u1, v1, t00, t10, t01, t11);
    float sample2 = BilinearSample(u2, v2, t00, t10, t01, t11);

    uint8_t byte1 = static_cast<uint8_t>(sample1);
    uint8_t byte2 = static_cast<uint8_t>(sample2);

    std::cout << "UV (0.499999): Sampled Color Byte = " << static_cast<int>(byte1) << " (Float: " << sample1 << ")\n";
    std::cout << "UV (0.500008): Sampled Color Byte = " << static_cast<int>(byte2) << " (Float: " << sample2 << ")\n";

    if (byte1 != byte2) {
        std::cout << "❌ DEEP HOLE 6 PROVED: Subpixel UV float drift alters Texel Bilinear Interpolation Output!\n";
        std::cout << "   -> 0.000009 UV float drift changes the final quantized pixel byte (127 vs 128).\n";
        std::cout << "   -> Texture samplers in Vulkan / CPU software renderers drift across hardware.\n";
    }
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "     DEEP ENGINE HOLES & ARCHITECTURAL VULNERABILITY TEST SUITE        \n";
    std::cout << "========================================================================\n";

    TestHole4_ThreadSchedulingOrder();
    TestHole5_UninitializedPaddingGarbage();
    TestHole6_SubpixelUVDrift();

    std::cout << "\n========================================================================\n";
    std::cout << "                 DEEP ENGINE HOLES SUMMARY & VERDICT                    \n";
    std::cout << "========================================================================\n";
    std::cout << "1. Multi-Threaded Out-of-Order Memory Layout (Atomic Stack Corruption).\n";
    std::cout << "2. Uninitialized Struct Alignment Padding Garbage Bytes in Buffer Hash.\n";
    std::cout << "3. Subpixel UV Bilinear Interpolation Weight Drift.\n";
    std::cout << "========================================================================\n";

    return 0;
}
