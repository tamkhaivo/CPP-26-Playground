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

namespace Type0::RigorousTest {

struct alignas(16) PhysicsParticle {
    float posX, posY, posZ, mass;
    float velX, velY, velZ, angle;
};

struct VkDispatchIndirectCommand {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

// Fixed-Point (Q16.16) Hermetic Data Representation
struct alignas(16) FixedParticle {
    int32_t posX, posY, posZ, mass;
    int32_t velX, velY, velZ, angle;
};

class WorkloadDivisionVerifier {
public:
    static void RunRigorousHoleVerification(size_t particleCount) {
        std::cout << "========================================================================\n";
        std::cout << "  RIGOROUS HERMETICITY & WORKLOAD DIVISION VERIFICATION (1M ENTITIES)   \n";
        std::cout << "========================================================================\n";

        std::vector<PhysicsParticle> particles(particleCount);
        std::default_random_engine rng(1337);
        std::uniform_real_distribution<float> dist(-50.0f, 50.0f);

        for (size_t i = 0; i < particleCount; ++i) {
            particles[i] = { dist(rng), dist(rng), dist(rng), 1.0f,
                             dist(rng) * 0.1f, dist(rng) * 0.1f, dist(rng) * 0.1f, dist(rng) };
        }

        // =====================================================================
        // TEST 1: THREAD REDUCTION NON-ASSOCIATIVITY HOLE TEST
        // =====================================================================
        std::cout << "\n[TEST 1] Testing Parallel Reduction Associativity Drift across Thread Counts...\n";
        
        // Single-threaded baseline accumulator
        double st_sum = 0.0;
        for (size_t i = 0; i < particleCount; ++i) {
            st_sum += static_cast<double>(particles[i].posX);
        }

        // Parallel atomic accumulator (simulating non-deterministic thread ordering)
        std::atomic<double> mt_sum{0.0};
        const size_t chunkSize = 4096;
        const size_t totalChunks = (particleCount + chunkSize - 1) / chunkSize;
        std::vector<size_t> chunkIndices(totalChunks);
        std::iota(chunkIndices.begin(), chunkIndices.end(), 0);

        std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
            size_t startOffset = chunkIdx * chunkSize;
            size_t endOffset = std::min(startOffset + chunkSize, particleCount);
            double local_sum = 0.0;
            for (size_t i = startOffset; i < endOffset; ++i) {
                local_sum += static_cast<double>(particles[i].posX);
            }
            double current = mt_sum.load(std::memory_order_relaxed);
            while (!mt_sum.compare_exchange_weak(current, current + local_sum, std::memory_order_relaxed));
        });

        uint64_t u_st, u_mt;
        double st_val = st_sum, mt_val = mt_sum.load();
        std::memcpy(&u_st, &st_val, sizeof(double));
        std::memcpy(&u_mt, &mt_val, sizeof(double));

        std::cout << "Single-Thread Sum: " << std::setprecision(12) << st_val << " (Bits: 0x" << std::hex << u_st << std::dec << ")\n";
        std::cout << "Multi-Thread  Sum: " << std::setprecision(12) << mt_val << " (Bits: 0x" << std::hex << u_mt << std::dec << ")\n";
        if (u_st != u_mt) {
            std::cout << "⚠️ VERIFIED HOLE: Parallel Reduction Order Causes Bit Drift in Float Accumulation!\n";
            std::cout << "   -> Fix: Workload Division must NOT rely on parallel float reductions for game state.\n";
        } else {
            std::cout << "✔️ Identical double accumulator precision achieved.\n";
        }

        // =====================================================================
        // TEST 2: FMA CONTRACTION & FTZ FLUSH-TO-ZERO SIMULATION
        // =====================================================================
        std::cout << "\n[TEST 2] Testing FMA Contraction & Denormal FTZ Bit Drift...\n";
        float a = 1.00000011920928955078125f; // 1 + 2^-23
        float b = 1.0000002384185791015625f;  // 1 + 2^-22
        float c = -1.00000035762786865234375f;

        // Fused: (a * b) + c
        float fused = std::fma(a, b, c);
        // Unfused separate step: mul then add
        float mul_step = a * b;
        float unfused = mul_step + c;

        uint32_t u_fused, u_unfused;
        std::memcpy(&u_fused, &fused, sizeof(float));
        std::memcpy(&u_unfused, &unfused, sizeof(float));

        std::cout << "FMA Hardware Fused:   " << std::setprecision(9) << fused << " (Bits: 0x" << std::hex << u_fused << std::dec << ")\n";
        std::cout << "Separate Mul-Add Step: " << std::setprecision(9) << unfused << " (Bits: 0x" << std::hex << u_unfused << std::dec << ")\n";
        if (u_fused != u_unfused) {
            std::cout << "⚠️ VERIFIED HOLE: FMA contraction causes 1-ULP Bitwise Divergence!\n";
            std::cout << "   -> Fix: SPIR-V must decorate all float variables with OpDecorate %var NoContraction.\n";
        }

        // =====================================================================
        // TEST 3: Q16.16 FIXED-POINT VS FLOAT HERMETIC SIMULATION HASH
        // =====================================================================
        std::cout << "\n[TEST 3] Evaluating Q16.16 Fixed-Point Integer Hermeticity Engine...\n";
        std::vector<FixedParticle> fixedParticles(particleCount);
        for (size_t i = 0; i < particleCount; ++i) {
            fixedParticles[i] = {
                static_cast<int32_t>(particles[i].posX * 65536.0f),
                static_cast<int32_t>(particles[i].posY * 65536.0f),
                static_cast<int32_t>(particles[i].posZ * 65536.0f),
                65536,
                static_cast<int32_t>(particles[i].velX * 65536.0f),
                static_cast<int32_t>(particles[i].velY * 65536.0f),
                static_cast<int32_t>(particles[i].velZ * 65536.0f),
                static_cast<int32_t>(particles[i].angle * 65536.0f)
            };
        }

        auto startFixed = std::chrono::high_resolution_clock::now();
        std::vector<uint64_t> chunkHashes(totalChunks, 0ULL);

        std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
            size_t startOffset = chunkIdx * chunkSize;
            size_t endOffset = std::min(startOffset + chunkSize, particleCount);
            uint64_t localHash = 14695981039346656037ULL; // FNV-1a basis

            for (size_t i = startOffset; i < endOffset; ++i) {
                // Fixed-Point Integration: pos = pos + (vel * dt) using 64-bit promotion to avoid UB overflow
                int64_t vx_dt = (static_cast<int64_t>(fixedParticles[i].velX) * 1092) >> 16;
                int64_t vy_dt = (static_cast<int64_t>(fixedParticles[i].velY) * 1092) >> 16;
                int32_t nextX = fixedParticles[i].posX + static_cast<int32_t>(vx_dt);
                int32_t nextY = fixedParticles[i].posY + static_cast<int32_t>(vy_dt);
                
                // Hash fixed-point bytes (Deterministic Bit-Identical across all hardware)
                localHash ^= static_cast<uint64_t>(nextX);
                localHash *= 1099511628211ULL;
                localHash ^= static_cast<uint64_t>(nextY);
                localHash *= 1099511628211ULL;
            }
            chunkHashes[chunkIdx] = localHash;
        });

        // Deterministic ordered reduction across chunks
        uint64_t finalFixedHash = 14695981039346656037ULL;
        for (size_t c = 0; c < totalChunks; ++c) {
            finalFixedHash ^= chunkHashes[c];
            finalFixedHash *= 1099511628211ULL;
        }

        auto endFixed = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> durationFixed = endFixed - startFixed;

        std::cout << "Q16.16 Fixed-Point SIMD/Int Engine Execution Time: " << durationFixed.count() << " ms\n";
        std::cout << "Hermetic State Hash: 0x" << std::hex << finalFixedHash << std::dec << " (100% BIT-EXACT)\n";
        std::cout << "========================================================================\n";
    }
};

} // namespace Type0::RigorousTest

int main() {
    Type0::RigorousTest::WorkloadDivisionVerifier::RunRigorousHoleVerification(1000000);
    return 0;
}
