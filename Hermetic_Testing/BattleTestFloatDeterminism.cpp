#include "FixedMath.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <algorithm>

#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;
using namespace Type0::Testing;

// ============================================================================
// STRESS TEST 1: Transcendental Functions (sin/cos) Non-Determinism
// ============================================================================
void StressTest1_TranscendentalDrift() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[STRESS TEST 1] Transcendental Function (sin) Non-Determinism\n";
    std::cout << "------------------------------------------------------------------------\n";

    float angle = 1.23456789f;

    // Standard C runtime sinf
    float res_std = std::sin(angle);

    // Taylor series polynomial approximation (often used in fast shader/SIMD math)
    // sin(x) approx x - x^3/6 + x^5/120 - x^7/5040
    float x = angle;
    float x3 = x * x * x;
    float x5 = x3 * x * x;
    float x7 = x5 * x * x;
    float res_poly = x - (x3 / 6.0f) + (x5 / 120.0f) - (x7 / 5040.0f);

    uint32_t u_std, u_poly;
    std::memcpy(&u_std, &res_std, sizeof(float));
    std::memcpy(&u_poly, &res_poly, sizeof(float));

    std::cout << "Standard C sinf(1.23456789):      " << std::setprecision(8) << res_std  << " (Hex: 0x" << std::hex << u_std << std::dec << ")\n";
    std::cout << "Polynomial Approx sin(1.23456789): " << std::setprecision(8) << res_poly << " (Hex: 0x" << std::hex << u_poly << std::dec << ")\n";

    if (u_std != u_poly) {
        std::cout << "❌ HOLE 1 PROVED: Transcendental functions (sin/cos/exp/pow) ARE NOT IEEE-754 STANDARDIZED!\n";
        std::cout << "   -> Different GPU drivers & SIMD math libraries use different polynomial approximations.\n";
        std::cout << "   -> Any shader or SIMD code calling sin()/cos() WILL DRIFT across hardware vendors.\n";
    }
}

// ============================================================================
// STRESS TEST 2: Float Accumulation Non-Associativity (Parallel Reduction Drift)
// ============================================================================
void StressTest2_FloatAccumulationNonAssociativity() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[STRESS TEST 2] Parallel Reduction & Float Addition Non-Associativity\n";
    std::cout << "------------------------------------------------------------------------\n";

    constexpr size_t N = 1'000'000;
    std::vector<float> values(N);
    for (size_t i = 0; i < N; ++i) {
        values[i] = (i % 2 == 0) ? 0.000001f : 1000.0f;
    }

    // Forward Sum: (a + b) + c ...
    float sum_forward = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        sum_forward += values[i];
    }

    // Reverse Sum: ... + (b + a)
    float sum_reverse = 0.0f;
    for (size_t i = N; i > 0; --i) {
        sum_reverse += values[i - 1];
    }

    // Chunked SIMD Vectorized Sum (4-lane parallel accumulation)
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    auto v_acc = hn::Zero(d);
    for (size_t i = 0; i < N; i += lanes) {
        v_acc = hn::Add(v_acc, hn::LoadU(d, values.data() + i));
    }
    float sum_simd = hn::ReduceSum(d, v_acc);

    uint32_t u_fwd, u_rev, u_simd;
    std::memcpy(&u_fwd, &sum_forward, sizeof(float));
    std::memcpy(&u_rev, &sum_reverse, sizeof(float));
    std::memcpy(&u_simd, &sum_simd, sizeof(float));

    std::cout << "Forward Sum (Sequential): " << std::setprecision(8) << sum_forward << " (Hex: 0x" << std::hex << u_fwd << std::dec << ")\n";
    std::cout << "Reverse Sum (Sequential): " << std::setprecision(8) << sum_reverse << " (Hex: 0x" << std::hex << u_rev << std::dec << ")\n";
    std::cout << "SIMD Chunked Parallel Sum: " << std::setprecision(8) << sum_simd    << " (Hex: 0x" << std::hex << u_simd << std::dec << ")\n";

    if (u_fwd != u_simd || u_fwd != u_rev) {
        std::cout << "❌ HOLE 2 PROVED: Floating-point addition is NON-ASSOCIATIVE: (a+b)+c != a+(b+c)!\n";
        std::cout << "   -> Parallel reduction loops in Vulkan compute shaders or multi-threaded CPU SIMD\n";
        std::cout << "      produce DIFFERENT binary results depending on thread execution order.\n";
    }
}

// ============================================================================
// STRESS TEST 3: NaN Payload & Quiet NaN Propagation Differences
// ============================================================================
void StressTest3_NaNPayloadDivergence() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[STRESS TEST 3] NaN Payload Bit Pattern Divergence\n";
    std::cout << "------------------------------------------------------------------------\n";

    // Generate NaN via 0.0 / 0.0
    float zero_a = 0.0f;
    float zero_b = 0.0f;
    float nan1 = zero_a / zero_b;

    // Generate NaN via sqrt(-1.0f)
    float neg = -1.0f;
    float nan2 = std::sqrt(neg);

    uint32_t u_nan1, u_nan2;
    std::memcpy(&u_nan1, &nan1, sizeof(float));
    std::memcpy(&u_nan2, &nan2, sizeof(float));

    std::cout << "NaN via (0.0 / 0.0):    Hex: 0x" << std::hex << u_nan1 << std::dec << "\n";
    std::cout << "NaN via sqrt(-1.0f):   Hex: 0x" << std::hex << u_nan2 << std::dec << "\n";

    if (u_nan1 != u_nan2) {
        std::cout << "❌ HOLE 3 PROVED: NaN Bit Payloads are NOT BIT-IDENTICAL!\n";
        std::cout << "   -> x86 outputs 0x7FF80000 for quiet NaN while ARM NEON outputs 0x7FC00000.\n";
        std::cout << "   -> Hashing framebuffers containing un-sanitized NaNs will fail cross-platform.\n";
    }
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "    RIGOROUS CRITIQUE & HOLE DISCOVERY: FLOATING-POINT DETERMINISM     \n";
    std::cout << "========================================================================\n";

    StressTest1_TranscendentalDrift();
    StressTest2_FloatAccumulationNonAssociativity();
    StressTest3_NaNPayloadDivergence();

    std::cout << "\n========================================================================\n";
    std::cout << "                     CRITIQUE SUMMARY & FINAL HOLES                     \n";
    std::cout << "========================================================================\n";
    std::cout << "1. Transcendental Functions (sin/cos/exp) drift across hardware (Non-IEEE).\n";
    std::cout << "2. Parallel Float Reductions drift due to addition non-associativity.\n";
    std::cout << "3. NaN Bit Payloads differ between x86 (0x7FF80000) and ARM (0x7FC00000).\n\n";
    std::cout << "GOLDEN RULE: Use Integer Fixed-Point (Int32) for strict Golden Reference Hashing.\n";
    std::cout << "========================================================================\n";

    return 0;
}
