// =============================================================================
// DirectPerformanceComparison.cpp — Rigorous Hermetic vs Non-Hermetic Benchmark
// Round 2: Addresses 6 methodological flaws from prior iteration.
// =============================================================================
//
// Fixes applied:
//   1. AVX2 enabled via CMake (/arch:AVX2) — expect 8 float32 lanes
//   2. True scalar baseline with optimization barrier (no auto-vectorization)
//   3. DoNotOptimize barrier via volatile sink + _ReadWriteBarrier
//   4. PRNG-generated non-repeating inputs via std::mt19937
//   5. Standard deviation reported alongside Min/Median/Mean
//   6. Explicit thread pool warmup before any benchmark measurement
//
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <execution>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <cstring>
#include <random>
#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>
#include <hwy/aligned_allocator.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace hn = hwy::HWY_NAMESPACE;

// =============================================================================
// Optimization Barrier — prevents compiler from eliminating or reordering work
// =============================================================================
static volatile float g_sink = 0.0f;

inline void DoNotOptimize(const float* data, size_t count) {
    // Touch last element through volatile to force materialization of all stores
    g_sink = data[count - 1];
#ifdef _MSC_VER
    _ReadWriteBarrier();
#else
    asm volatile("" ::: "memory");
#endif
}

// =============================================================================
// 64-bit FNV-1a Hash for determinism verification
// =============================================================================
uint64_t ComputeChecksum(const float* data, size_t count) {
    uint64_t hash = 14695981039346656037ULL;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    const size_t totalBytes = count * sizeof(float);
    for (size_t i = 0; i < totalBytes; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// =============================================================================
// Statistical benchmark harness with warmup, multi-iteration, and stddev
// =============================================================================
struct Stats {
    double min_ms   = 0.0;
    double max_ms   = 0.0;
    double mean_ms  = 0.0;
    double median_ms = 0.0;
    double stddev_ms = 0.0;
    uint64_t checksum = 0;
};

template<typename Func>
Stats RunBenchmark(const char* name, Func&& func, int warmup = 5, int runs = 15) {
    std::cout << "--> [" << name << "] (" << warmup << " warmups, " << runs << " eval runs)\n";

    // Warmup passes — bring caches, branch predictors, and frequency scaling to steady state
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    std::vector<double> timings;
    timings.reserve(runs);
    uint64_t last_hash = 0;

    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        last_hash = func();
        auto t1 = std::chrono::high_resolution_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(timings.begin(), timings.end());
    Stats s;
    s.min_ms    = timings.front();
    s.max_ms    = timings.back();
    s.median_ms = timings[runs / 2];
    s.mean_ms   = std::accumulate(timings.begin(), timings.end(), 0.0) / runs;
    s.checksum  = last_hash;

    // Standard deviation
    double variance = 0.0;
    for (double t : timings) {
        double diff = t - s.mean_ms;
        variance += diff * diff;
    }
    s.stddev_ms = std::sqrt(variance / runs);

    std::cout << "    Min: " << std::fixed << std::setprecision(2) << s.min_ms
              << " | Median: " << s.median_ms
              << " | Mean: " << s.mean_ms
              << " | StdDev: " << std::setprecision(3) << s.stddev_ms
              << " ms | Hash: 0x" << std::hex << s.checksum << std::dec << "\n\n";

    return s;
}

// =============================================================================
// TRUE SCALAR BASELINE — optimization disabled, no auto-vectorization possible
// =============================================================================
#pragma optimize("", off)
#pragma float_control(precise, on, push)
void ScalarTranscendentalLoop(const float* __restrict in, float* __restrict out, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        float x = in[i];
        out[i] = std::sin(x) + std::cos(x) * std::exp(x * 0.01f) + std::log(x + 1.0f);
    }
}
#pragma float_control(pop)
#pragma optimize("", on)

// =============================================================================
// MAIN BENCHMARK
// =============================================================================
int main() {
    std::cout << "====================================================================================================================\n";
    std::cout << "        RIGOROUS BENCHMARK v2: OPTIMAL NON-HERMETIC VS OPTIMAL HERMETIC (6-FLAW CRITIQUE APPLIED)                  \n";
    std::cout << "====================================================================================================================\n\n";

    // --- System Info ---
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    std::cout << "[Compiler]          : MSVC " << _MSC_VER << "\n";
    std::cout << "[SIMD Lanes]        : " << lanes << " float32 (";
    if (lanes >= 16) std::cout << "AVX-512";
    else if (lanes >= 8) std::cout << "AVX2";
    else if (lanes >= 4) std::cout << "SSE2/SSE4";
    else std::cout << "Scalar";
    std::cout << " detected)\n";

    constexpr size_t N = 20'000'000;
    std::cout << "[Dataset]           : " << N << " elements (" << (N * sizeof(float)) / (1024 * 1024) << " MB)\n";
    std::cout << "[Warmup Iterations] : 5\n";
    std::cout << "[Eval Iterations]   : 15 (reporting Min/Median/Mean/StdDev)\n";
    std::cout << "[Input Distribution]: std::mt19937 PRNG, uniform [0.001, 10.0]\n\n";

    // --- PRNG Input Generation (Flaw 4 fix) ---
    hwy::AlignedVector<float> inputs(N);
    hwy::AlignedVector<float> out_scalar(N);
    hwy::AlignedVector<float> out_nh_auto(N);
    hwy::AlignedVector<float> out_nh_multi(N);
    hwy::AlignedVector<float> out_h_single(N);
    hwy::AlignedVector<float> out_h_multi(N);

    {
        std::mt19937 rng(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(0.001f, 10.0f);
        for (size_t i = 0; i < N; ++i) {
            inputs[i] = dist(rng);
        }
    }

    // --- Thread Pool Warmup (Flaw 6 fix) ---
    {
        std::cout << "[Thread Pool Warmup]: Priming std::execution::par thread pool...\n\n";
        std::vector<size_t> dummy(1024);
        std::iota(dummy.begin(), dummy.end(), 0);
        std::for_each(std::execution::par, dummy.begin(), dummy.end(), [](size_t& x) {
            volatile size_t sink = x * x;
            (void)sink;
        });
    }

    // --- Chunk setup for parallel tests ---
    constexpr size_t chunkSize = 32768;
    const size_t totalChunks = (N + chunkSize - 1) / chunkSize;
    std::vector<size_t> chunkIndices(totalChunks);
    std::iota(chunkIndices.begin(), chunkIndices.end(), 0);

    // =========================================================================
    // TEST 1: True Scalar Baseline (optimization disabled, no auto-vectorization)
    // =========================================================================
    auto stats_scalar = RunBenchmark("1. TRUE SCALAR (optimization off, no auto-vec)", [&]() -> uint64_t {
        ScalarTranscendentalLoop(inputs.data(), out_scalar.data(), N);
        DoNotOptimize(out_scalar.data(), N);
        return ComputeChecksum(out_scalar.data(), N);
    });

    // =========================================================================
    // TEST 2: Non-Hermetic Single-Threaded (MSVC /O2 auto-vectorization allowed)
    // =========================================================================
    auto stats_nh_auto = RunBenchmark("2. Non-Hermetic ST (MSVC /O2, auto-vec allowed)", [&]() -> uint64_t {
        for (size_t i = 0; i < N; ++i) {
            float x = inputs[i];
            out_nh_auto[i] = std::sin(x) + std::cos(x) * std::exp(x * 0.01f) + std::log(x + 1.0f);
        }
        DoNotOptimize(out_nh_auto.data(), N);
        return ComputeChecksum(out_nh_auto.data(), N);
    });

    // =========================================================================
    // TEST 3: Non-Hermetic Multi-Threaded (MSVC /O2 + std::execution::par)
    // =========================================================================
    auto stats_nh_multi = RunBenchmark("3. Non-Hermetic MT (MSVC /O2 + par)", [&]() -> uint64_t {
        std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
            size_t start = chunkIdx * chunkSize;
            size_t end = std::min(start + chunkSize, N);
            for (size_t i = start; i < end; ++i) {
                float x = inputs[i];
                out_nh_multi[i] = std::sin(x) + std::cos(x) * std::exp(x * 0.01f) + std::log(x + 1.0f);
            }
        });
        DoNotOptimize(out_nh_multi.data(), N);
        return ComputeChecksum(out_nh_multi.data(), N);
    });

    // =========================================================================
    // TEST 4: Hermetic Single-Threaded SIMD (Google Highway AVX2)
    // =========================================================================
    const auto v_one   = hn::Set(d, 1.0f);
    const auto v_scale = hn::Set(d, 0.01f);

    auto stats_h_single = RunBenchmark("4. Hermetic ST (Highway SIMD, explicit vectorization)", [&]() -> uint64_t {
        for (size_t i = 0; i < N; i += lanes) {
            auto vx    = hn::Load(d, inputs.data() + i);
            auto v_sin = hn::Sin(d, vx);
            auto v_cos = hn::Cos(d, vx);
            auto v_exp = hn::Exp(d, hn::Mul(vx, v_scale));
            auto v_log = hn::Log(d, hn::Add(vx, v_one));
            auto vr    = hn::Add(v_sin, hn::Add(hn::Mul(v_cos, v_exp), v_log));
            hn::Store(vr, d, out_h_single.data() + i);
        }
        DoNotOptimize(out_h_single.data(), N);
        return ComputeChecksum(out_h_single.data(), N);
    });

    // =========================================================================
    // TEST 5: Hermetic Multi-Threaded SIMD (Highway AVX2 + std::execution::par)
    // =========================================================================
    auto stats_h_multi = RunBenchmark("5. Hermetic MT (Highway SIMD + par)", [&]() -> uint64_t {
        std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
            size_t start = chunkIdx * chunkSize;
            size_t end = std::min(start + chunkSize, N);
            for (size_t i = start; i < end; i += lanes) {
                auto vx    = hn::Load(d, inputs.data() + i);
                auto v_sin = hn::Sin(d, vx);
                auto v_cos = hn::Cos(d, vx);
                auto v_exp = hn::Exp(d, hn::Mul(vx, v_scale));
                auto v_log = hn::Log(d, hn::Add(vx, v_one));
                auto vr    = hn::Add(v_sin, hn::Add(hn::Mul(v_cos, v_exp), v_log));
                hn::Store(vr, d, out_h_multi.data() + i);
            }
        });
        DoNotOptimize(out_h_multi.data(), N);
        return ComputeChecksum(out_h_multi.data(), N);
    });

    // =========================================================================
    // PERFORMANCE COMPARISON MATRIX
    // =========================================================================
    auto PrintRow = [&](const char* label, const Stats& s, double baseline_st, double baseline_mt) {
        double ns_per = s.median_ms * 1e6 / N;
        std::cout << "  " << std::left << std::setw(47) << label
                  << "| " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << s.median_ms << " ms"
                  << " | " << std::setw(5) << std::setprecision(3) << s.stddev_ms << " ms"
                  << " | " << std::setw(7) << std::setprecision(2) << ns_per << " ns/e"
                  << " | " << std::setw(6) << std::setprecision(2) << (baseline_st / s.median_ms) << "x"
                  << " | ";
        if (baseline_mt > 0.0)
            std::cout << std::setw(6) << std::setprecision(2) << (baseline_mt / s.median_ms) << "x";
        else
            std::cout << "   -- ";
        std::cout << "\n";
    };

    std::cout << "====================================================================================================================\n";
    std::cout << "                          RIGOROUS PERFORMANCE MATRIX (v2 — 6 FLAWS FIXED)                                         \n";
    std::cout << "====================================================================================================================\n";
    std::cout << "  Pipeline                                       | Median    | StdDev   | Latency   | vs Scalar | vs NH-MT\n";
    std::cout << "--------------------------------------------------------------------------------------------------------------------\n";
    PrintRow("1. True Scalar (opt-off, guaranteed no SIMD)",      stats_scalar,    stats_scalar.median_ms, -1.0);
    PrintRow("2. Non-Hermetic ST (MSVC /O2, auto-vec possible)", stats_nh_auto,   stats_scalar.median_ms, -1.0);
    PrintRow("3. Non-Hermetic MT (MSVC /O2 + par)",              stats_nh_multi,  stats_scalar.median_ms, stats_nh_multi.median_ms);
    PrintRow("4. Hermetic ST (Highway SIMD)",                    stats_h_single,  stats_scalar.median_ms, -1.0);
    PrintRow("5. Hermetic MT (Highway SIMD + par)",              stats_h_multi,   stats_scalar.median_ms, stats_nh_multi.median_ms);
    std::cout << "====================================================================================================================\n";

    // =========================================================================
    // DETERMINISM AUDIT
    // =========================================================================
    std::cout << "\n[Determinism Checksum Audit]\n";
    std::cout << "  True Scalar Hash    : 0x" << std::hex << stats_scalar.checksum << std::dec << "\n";
    std::cout << "  Non-Hermetic ST Hash: 0x" << std::hex << stats_nh_auto.checksum << std::dec << "\n";
    std::cout << "  Non-Hermetic MT Hash: 0x" << std::hex << stats_nh_multi.checksum << std::dec << "\n";
    std::cout << "  Hermetic ST Hash    : 0x" << std::hex << stats_h_single.checksum << std::dec << "\n";
    std::cout << "  Hermetic MT Hash    : 0x" << std::hex << stats_h_multi.checksum << std::dec << "\n\n";

    // Cross-check determinism
    if (stats_scalar.checksum == stats_nh_auto.checksum)
        std::cout << "  [Scalar vs NH-ST]   : IDENTICAL (MSVC did NOT auto-vectorize, or vectorized path is bit-exact)\n";
    else
        std::cout << "  [Scalar vs NH-ST]   : DIVERGENT — MSVC auto-vectorization changed floating-point bit pattern!\n";

    if (stats_nh_auto.checksum == stats_nh_multi.checksum)
        std::cout << "  [NH-ST vs NH-MT]    : IDENTICAL\n";
    else
        std::cout << "  [NH-ST vs NH-MT]    : DIVERGENT — threading changed FP evaluation order!\n";

    if (stats_h_single.checksum == stats_h_multi.checksum)
        std::cout << "  [Hermetic ST vs MT] : ✅ IDENTICAL — 100% BIT-FOR-BIT DETERMINISTIC across threads!\n";
    else
        std::cout << "  [Hermetic ST vs MT] : ❌ DIVERGENT — unexpected!\n";

    if (stats_h_single.checksum != stats_nh_auto.checksum)
        std::cout << "  [Hermetic vs NH]    : EXPECTED DIVERGENCE — different polynomial approximations (hermetic minimax vs OS libm)\n";

    std::cout << "\n====================================================================================================================\n";
    std::cout << "                                          BENCHMARK COMPLETE                                                        \n";
    std::cout << "====================================================================================================================\n";

    return 0;
}
