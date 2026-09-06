#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <execution>
#include <chrono>
#include <random>
#include <atomic>

// Google Highway SIMD
#include <hwy/highway.h>

namespace Type0 {

// Struct layout matching Vulkan 1.4 storage buffer std430 alignment
struct alignas(16) AABB {
    float minX, minY, minZ, pad0;
    float maxX, maxY, maxZ, pad1;
};

// Represents Vulkan 1.4 Indirect Dispatch Command structure
struct VkDispatchIndirectCommand {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

// SIMD Accelerated CPU Broadphase & Dynamic Vulkan Dispatch Generator
class CPUBroadphasePipeline {
public:
    static void RunCullingBenchmark(size_t objectCount) {
        std::cout << "=== Running CPU SIMD Broadphase & Vulkan Packing Benchmark ===" << std::endl;
        std::cout << "Object Count: " << objectCount << std::endl;

        std::vector<AABB> boxes(objectCount);
        std::default_random_engine rng(42);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

        // Generate synthetic AABBs
        for (size_t i = 0; i < objectCount; ++i) {
            float cx = dist(rng), cy = dist(rng), cz = dist(rng);
            boxes[i] = { cx - 1.0f, cy - 1.0f, cz - 1.0f, 0.0f, cx + 1.0f, cy + 1.0f, cz + 1.0f, 0.0f };
        }

        // Bounding Volume Culling Box (e.g. Frustum / Active Grid Region)
        AABB queryRegion = { -10.0f, -10.0f, -10.0f, 0.0f, 10.0f, 10.0f, 10.0f, 0.0f };

        // Output vector storing dynamic indices
        std::vector<uint32_t> visibleIndices(objectCount);
        std::atomic<size_t> visibleCount{0};

        const size_t chunkSize = 4096;
        const size_t totalChunks = (objectCount + chunkSize - 1) / chunkSize;
        std::vector<size_t> chunkIndices(totalChunks);
        std::iota(chunkIndices.begin(), chunkIndices.end(), 0);

        auto startTime = std::chrono::high_resolution_clock::now();

        // Multi-threaded CPU execution with parallel algorithm
        std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
            size_t startOffset = chunkIdx * chunkSize;
            size_t endOffset = std::min(startOffset + chunkSize, objectCount);

            namespace hn = hwy::HWY_NAMESPACE;
            const hn::ScalableTag<float> d;
            const size_t lanes = hn::Lanes(d);

            const auto qMinX = hn::Set(d, queryRegion.minX);
            const auto qMinY = hn::Set(d, queryRegion.minY);
            const auto qMaxX = hn::Set(d, queryRegion.maxX);
            const auto qMaxY = hn::Set(d, queryRegion.maxY);
            const auto qMinZ = hn::Set(d, queryRegion.minZ);
            const auto qMaxZ = hn::Set(d, queryRegion.maxZ);

            size_t i = startOffset;
            for (; i + lanes <= endOffset; i += lanes) {
                alignas(16) float bMinX[16], bMinY[16], bMinZ[16];
                alignas(16) float bMaxX[16], bMaxY[16], bMaxZ[16];

                for (size_t l = 0; l < lanes; ++l) {
                    bMinX[l] = boxes[i + l].minX;
                    bMinY[l] = boxes[i + l].minY;
                    bMinZ[l] = boxes[i + l].minZ;
                    bMaxX[l] = boxes[i + l].maxX;
                    bMaxY[l] = boxes[i + l].maxY;
                    bMaxZ[l] = boxes[i + l].maxZ;
                }

                auto boxMinX = hn::Load(d, bMinX);
                auto boxMinY = hn::Load(d, bMinY);
                auto boxMinZ = hn::Load(d, bMinZ);
                auto boxMaxX = hn::Load(d, bMaxX);
                auto boxMaxY = hn::Load(d, bMaxY);
                auto boxMaxZ = hn::Load(d, bMaxZ);

                // AABB Overlap Condition: (box.min <= query.max) && (box.max >= query.min)
                auto overlapX = hn::And(hn::Le(boxMinX, qMaxX), hn::Ge(boxMaxX, qMinX));
                auto overlapY = hn::And(hn::Le(boxMinY, qMaxY), hn::Ge(boxMaxY, qMinY));
                auto overlapZ = hn::And(hn::Le(boxMinZ, qMaxZ), hn::Ge(boxMaxZ, qMinZ));
                auto overlap = hn::And(hn::And(overlapX, overlapY), overlapZ);

                // Check if any lane matched
                if (!hn::AllFalse(d, overlap)) {
                    // Convert mask to vector via IfThenElse
                    auto vOverlap = hn::IfThenElse(overlap, hn::Set(d, 1.0f), hn::Set(d, 0.0f));
                    alignas(16) float res[16];
                    hn::Store(vOverlap, d, res);

                    for (size_t l = 0; l < lanes; ++l) {
                        if (res[l] > 0.5f) {
                            size_t idx = visibleCount.fetch_add(1, std::memory_order_relaxed);
                            visibleIndices[idx] = static_cast<uint32_t>(i + l);
                        }
                    }
                }
            }

            // Scalar Tail Fallback
            for (; i < endOffset; ++i) {
                if (boxes[i].minX <= queryRegion.maxX && boxes[i].maxX >= queryRegion.minX &&
                    boxes[i].minY <= queryRegion.maxY && boxes[i].maxY >= queryRegion.minY &&
                    boxes[i].minZ <= queryRegion.maxZ && boxes[i].maxZ >= queryRegion.minZ) {
                    size_t idx = visibleCount.fetch_add(1, std::memory_order_relaxed);
                    visibleIndices[idx] = static_cast<uint32_t>(i);
                }
            }
        });

        // Compute Vulkan 1.4 Indirect Dispatch Structure from CPU output
        uint32_t totalVisible = static_cast<uint32_t>(visibleCount.load());
        VkDispatchIndirectCommand indirectCmd{};
        indirectCmd.x = (totalVisible + 63) / 64; // Workgroup size = 64
        indirectCmd.y = 1;
        indirectCmd.z = 1;

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = endTime - startTime;

        std::cout << "Completed in: " << duration.count() << " ms" << std::endl;
        std::cout << "Total Visible Entities Found: " << totalVisible << std::endl;
        std::cout << "Calculated Vulkan Indirect Dispatch Group (x, y, z): ("
                  << indirectCmd.x << ", " << indirectCmd.y << ", " << indirectCmd.z << ")" << std::endl;
    }
};

} // namespace Type0

int main() {
    Type0::CPUBroadphasePipeline::RunCullingBenchmark(500000);
    return 0;
}
