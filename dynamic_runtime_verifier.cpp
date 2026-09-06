#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <execution>
#include <chrono>
#include <random>
#include <atomic>
#include <iomanip>
#include <cstring>
#include <cmath>

#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>

namespace Type0::DynamicTest {

namespace hn = hwy::HWY_NAMESPACE;

struct alignas(16) Entity {
    float px, py, pz, mass;
    float vx, vy, vz, radius;
};

struct DynamicMetrics {
    double totalTimeMs;
    size_t activeCount;
    uint32_t indirectX;
    uint64_t fnv1aHash;
};

class DynamicHarness {
public:
    static DynamicMetrics ExecuteDynamicTest(size_t entityCount, bool enforceStrictHermetic) {
        std::vector<Entity> entities(entityCount);
        std::default_random_engine rng(42);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

        for (size_t i = 0; i < entityCount; ++i) {
            entities[i] = { dist(rng), dist(rng), dist(rng), 1.0f,
                            dist(rng) * 0.05f, dist(rng) * 0.05f, dist(rng) * 0.05f, 1.0f };
        }

        std::vector<uint32_t> candidateIndices(entityCount);

        const size_t chunkSize = 4096;
        const size_t totalChunks = (entityCount + chunkSize - 1) / chunkSize;
        std::vector<size_t> chunkIndices(totalChunks);
        std::iota(chunkIndices.begin(), chunkIndices.end(), 0);

        // Track per-chunk counts to eliminate thread atomic ordering non-determinism
        std::vector<size_t> chunkCounts(totalChunks, 0);
        std::vector<size_t> chunkOffsets(totalChunks, 0);

        auto start = std::chrono::high_resolution_clock::now();

        // Pass 1: Parallel spatial broadphase and per-chunk count gathering
        std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
            size_t startOffset = chunkIdx * chunkSize;
            size_t endOffset = std::min(startOffset + chunkSize, entityCount);

            const hn::ScalableTag<float> d;
            const size_t lanes = hn::Lanes(d);

            const auto boundsMin = hn::Set(d, -50.0f);
            const auto boundsMax = hn::Set(d, 50.0f);

            size_t localCount = 0;
            size_t i = startOffset;
            for (; i + lanes <= endOffset; i += lanes) {
                alignas(16) float x[16], y[16], z[16];
                for (size_t l = 0; l < lanes; ++l) {
                    x[l] = entities[i + l].px;
                    y[l] = entities[i + l].py;
                    z[l] = entities[i + l].pz;
                }

                auto vx = hn::Load(d, x);
                auto vy = hn::Load(d, y);
                auto vz = hn::Load(d, z);

                if (enforceStrictHermetic) {
                    const auto expMask = hn::BitCast(d, hn::Set(hn::ScalableTag<uint32_t>(), 0x7F800000u));
                    vx = hn::And(vx, expMask);
                    vy = hn::And(vy, expMask);
                    vz = hn::And(vz, expMask);
                }

                auto inX = hn::And(hn::Ge(vx, boundsMin), hn::Le(vx, boundsMax));
                auto inY = hn::And(hn::Ge(vy, boundsMin), hn::Le(vy, boundsMax));
                auto inZ = hn::And(hn::Ge(vz, boundsMin), hn::Le(vz, boundsMax));
                auto valid = hn::And(hn::And(inX, inY), inZ);

                if (!hn::AllFalse(d, valid)) {
                    auto vRes = hn::IfThenElse(valid, hn::Set(d, 1.0f), hn::Set(d, 0.0f));
                    alignas(16) float res[16];
                    hn::Store(vRes, d, res);

                    for (size_t l = 0; l < lanes; ++l) {
                        if (res[l] > 0.5f) {
                            localCount++;
                        }
                    }
                }
            }

            for (; i < endOffset; ++i) {
                if (entities[i].px >= -50.0f && entities[i].px <= 50.0f &&
                    entities[i].py >= -50.0f && entities[i].py <= 50.0f &&
                    entities[i].pz >= -50.0f && entities[i].pz <= 50.0f) {
                    localCount++;
                }
            }
            chunkCounts[chunkIdx] = localCount;
        });

        // Prefix sum for deterministic index offsets across chunks
        size_t totalActive = 0;
        for (size_t c = 0; c < totalChunks; ++c) {
            chunkOffsets[c] = totalActive;
            totalActive += chunkCounts[c];
        }

        // Pass 2: Parallel writing into deterministic pre-calculated offsets
        std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
            size_t startOffset = chunkIdx * chunkSize;
            size_t endOffset = std::min(startOffset + chunkSize, entityCount);
            size_t writeIdx = chunkOffsets[chunkIdx];

            const hn::ScalableTag<float> d;
            const size_t lanes = hn::Lanes(d);

            const auto boundsMin = hn::Set(d, -50.0f);
            const auto boundsMax = hn::Set(d, 50.0f);

            size_t i = startOffset;
            for (; i + lanes <= endOffset; i += lanes) {
                alignas(16) float x[16], y[16], z[16];
                for (size_t l = 0; l < lanes; ++l) {
                    x[l] = entities[i + l].px;
                    y[l] = entities[i + l].py;
                    z[l] = entities[i + l].pz;
                }

                auto vx = hn::Load(d, x);
                auto vy = hn::Load(d, y);
                auto vz = hn::Load(d, z);

                if (enforceStrictHermetic) {
                    const auto expMask = hn::BitCast(d, hn::Set(hn::ScalableTag<uint32_t>(), 0x7F800000u));
                    vx = hn::And(vx, expMask);
                    vy = hn::And(vy, expMask);
                    vz = hn::And(vz, expMask);
                }

                auto inX = hn::And(hn::Ge(vx, boundsMin), hn::Le(vx, boundsMax));
                auto inY = hn::And(hn::Ge(vy, boundsMin), hn::Le(vy, boundsMax));
                auto inZ = hn::And(hn::Ge(vz, boundsMin), hn::Le(vz, boundsMax));
                auto valid = hn::And(hn::And(inX, inY), inZ);

                if (!hn::AllFalse(d, valid)) {
                    auto vRes = hn::IfThenElse(valid, hn::Set(d, 1.0f), hn::Set(d, 0.0f));
                    alignas(16) float res[16];
                    hn::Store(vRes, d, res);

                    for (size_t l = 0; l < lanes; ++l) {
                        if (res[l] > 0.5f) {
                            candidateIndices[writeIdx++] = static_cast<uint32_t>(i + l);
                        }
                    }
                }
            }

            for (; i < endOffset; ++i) {
                if (entities[i].px >= -50.0f && entities[i].px <= 50.0f &&
                    entities[i].py >= -50.0f && entities[i].py <= 50.0f &&
                    entities[i].pz >= -50.0f && entities[i].pz <= 50.0f) {
                    candidateIndices[writeIdx++] = static_cast<uint32_t>(i);
                }
            }
        });

        auto end = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

        // Compute FNV-1a state hash over deterministically ordered active candidates
        uint64_t hash = 14695981039346656037ULL;
        for (size_t k = 0; k < totalActive; ++k) {
            hash ^= candidateIndices[k];
            hash *= 1099511628211ULL;
        }

        uint32_t dispatchX = (static_cast<uint32_t>(totalActive) + 63) / 64;

        return DynamicMetrics{ elapsedMs, totalActive, dispatchX, hash };
    }
};

} // namespace Type0::DynamicTest

int main() {
    std::cout << "========================================================================\n";
    std::cout << "  DYNAMIC VERIFICATION & PROOF: DETERMINISTIC INDEX PREFIX SUM FIX      \n";
    std::cout << "========================================================================\n\n";

    std::vector<size_t> testScales = { 100'000, 500'000, 1'000'000, 2'000'000 };

    for (size_t scale : testScales) {
        std::cout << "--> Scaling Benchmark: " << scale << " Physics Entities\n";

        // Run 1: Execution Pass A
        auto run1 = Type0::DynamicTest::DynamicHarness::ExecuteDynamicTest(scale, true);
        // Run 2: Execution Pass B (Repeatability Check)
        auto run2 = Type0::DynamicTest::DynamicHarness::ExecuteDynamicTest(scale, true);

        std::cout << "    Execution Time:                 " << std::fixed << std::setprecision(4) << run1.totalTimeMs << " ms\n";
        std::cout << "    Filtered Active Candidates:      " << run1.activeCount << "\n";
        std::cout << "    Vulkan 1.4 Indirect Workgroups:  " << run1.indirectX << " (x, 1, 1)\n";
        std::cout << "    Run 1 State Hash:               0x" << std::hex << run1.fnv1aHash << std::dec << "\n";
        std::cout << "    Run 2 State Hash:               0x" << std::hex << run2.fnv1aHash << std::dec << "\n";

        if (run1.fnv1aHash == run2.fnv1aHash) {
            std::cout << "    Verification Status:            [PASS] 100% BIT-EXACT REPEATABLE!\n";
        } else {
            std::cout << "    Verification Status:            [FAIL] DIVERGENCE DETECTED\n";
        }
        std::cout << "------------------------------------------------------------------------\n";
    }

    return 0;
}
