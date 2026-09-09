#include "FixedMath.hpp"
#include "ImageMetrics.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cassert>

// Include Google Highway header
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;
using namespace Type0::Testing;

// ============================================================================
// 1. Scalar Reference Edge Evaluation (Fixed-Point 16.16)
// ============================================================================
void ScalarEdgeFunctionFixedBatch(
    const int32_t* __restrict px_raw,
    const int32_t* __restrict py_raw,
    int32_t ax, int32_t ay, int32_t bx, int32_t by,
    int32_t* __restrict out_w,
    size_t count)
{
    // Shift down 8 bits prior to multiplication so 32-bit math doesn't overflow
    int32_t dx = (bx - ax) >> 8;
    int32_t dy = (by - ay) >> 8;

    for (size_t i = 0; i < count; ++i) {
        int32_t sub_x = (px_raw[i] - ax) >> 8;
        int32_t sub_y = (py_raw[i] - ay) >> 8;
        int32_t res = (sub_x * dy - sub_y * dx);
        out_w[i] = res;
    }
}

// ============================================================================
// 2. Naive Floating-Point SIMD (Highway Float SIMD)
// ============================================================================
void HwyFloatEdgeFunctionBatch(
    const float* __restrict px,
    const float* __restrict py,
    float ax, float ay, float bx, float by,
    float* __restrict out_w,
    size_t count)
{
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    const auto v_ax = hn::Set(d, ax);
    const auto v_ay = hn::Set(d, ay);
    const auto v_dx = hn::Set(d, bx - ax);
    const auto v_dy = hn::Set(d, by - ay);

    for (size_t i = 0; i < count; i += lanes) {
        auto v_px = hn::LoadU(d, px + i);
        auto v_py = hn::LoadU(d, py + i);

        auto sub_x = hn::Sub(v_px, v_ax);
        auto sub_y = hn::Sub(v_py, v_ay);

        auto term1 = hn::Mul(sub_x, v_dy);
        auto term2 = hn::Mul(sub_y, v_dx);
        auto w = hn::Sub(term1, term2);

        hn::StoreU(w, d, out_w + i);
    }
}

// ============================================================================
// 3. Vectorized Deterministic Fixed-Point SIMD (Highway Int32 SIMD)
// ============================================================================
void HwyFixedPointEdgeFunctionBatch(
    const int32_t* __restrict px_raw,
    const int32_t* __restrict py_raw,
    int32_t ax, int32_t ay, int32_t bx, int32_t by,
    int32_t* __restrict out_w,
    size_t count)
{
    const hn::ScalableTag<int32_t> d;
    const size_t lanes = hn::Lanes(d);

    const auto v_ax = hn::Set(d, ax);
    const auto v_ay = hn::Set(d, ay);
    const auto v_dx = hn::ShiftRight<8>(hn::Set(d, bx - ax));
    const auto v_dy = hn::ShiftRight<8>(hn::Set(d, by - ay));

    for (size_t i = 0; i < count; i += lanes) {
        auto v_px = hn::LoadU(d, px_raw + i);
        auto v_py = hn::LoadU(d, py_raw + i);

        auto sub_x = hn::ShiftRight<8>(hn::Sub(v_px, v_ax));
        auto sub_y = hn::ShiftRight<8>(hn::Sub(v_py, v_ay));

        auto term1 = hn::Mul(sub_x, v_dy);
        auto term2 = hn::Mul(sub_y, v_dx);
        auto w_fixed = hn::Sub(term1, term2);

        hn::StoreU(w_fixed, d, out_w + i);
    }
}

// ============================================================================
// Benchmark & Determinism Test Runner
// ============================================================================
int main() {
    constexpr size_t NUM_PIXELS = 1'000'000;
    std::cout << "Preparing dataset of " << NUM_PIXELS << " pixels for benchmark...\n";

    std::vector<float> px_float(NUM_PIXELS);
    std::vector<float> py_float(NUM_PIXELS);
    std::vector<int32_t> px_int(NUM_PIXELS);
    std::vector<int32_t> py_int(NUM_PIXELS);

    for (size_t i = 0; i < NUM_PIXELS; ++i) {
        float x = static_cast<float>(i % 1000) + 0.5f;
        float y = static_cast<float>(i / 1000) + 0.5f;
        px_float[i] = x;
        py_float[i] = y;
        px_int[i] = Fixed16(x).v;
        py_int[i] = Fixed16(y).v;
    }

    float ax_f = 0.5f, ay_f = 0.5f;
    float bx_f = 999.5f, by_f = 12.5f;

    int32_t ax_i = Fixed16(ax_f).v, ay_i = Fixed16(ay_f).v;
    int32_t bx_i = Fixed16(bx_f).v, by_i = Fixed16(by_f).v;

    std::vector<int32_t> scalar_results(NUM_PIXELS);
    std::vector<float> hwy_float_results(NUM_PIXELS);
    std::vector<int32_t> hwy_fixed_results(NUM_PIXELS);

    // Warm-up cache
    ScalarEdgeFunctionFixedBatch(px_int.data(), py_int.data(), ax_i, ay_i, bx_i, by_i, scalar_results.data(), NUM_PIXELS);

    // 1. Benchmark Scalar Fixed-Point
    auto start_scalar = std::chrono::high_resolution_clock::now();
    ScalarEdgeFunctionFixedBatch(px_int.data(), py_int.data(), ax_i, ay_i, bx_i, by_i, scalar_results.data(), NUM_PIXELS);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double time_scalar_ms = std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count();

    // 2. Benchmark Highway Float SIMD
    auto start_hwy_float = std::chrono::high_resolution_clock::now();
    HwyFloatEdgeFunctionBatch(px_float.data(), py_float.data(), ax_f, ay_f, bx_f, by_f, hwy_float_results.data(), NUM_PIXELS);
    auto end_hwy_float = std::chrono::high_resolution_clock::now();
    double time_hwy_float_ms = std::chrono::duration<double, std::milli>(end_hwy_float - start_hwy_float).count();

    // 3. Benchmark Highway Fixed-Point SIMD (Deterministic Path)
    auto start_hwy_fixed = std::chrono::high_resolution_clock::now();
    HwyFixedPointEdgeFunctionBatch(px_int.data(), py_int.data(), ax_i, ay_i, bx_i, by_i, hwy_fixed_results.data(), NUM_PIXELS);
    auto end_hwy_fixed = std::chrono::high_resolution_clock::now();
    double time_hwy_fixed_ms = std::chrono::duration<double, std::milli>(end_hwy_fixed - start_hwy_fixed).count();

    // Discrepancy Check
    size_t float_discrepancies = 0;
    size_t fixed_discrepancies = 0;

    for (size_t i = 0; i < NUM_PIXELS; ++i) {
        int32_t scalar_val = scalar_results[i];
        
        // Convert float result back to fixed space (16 fractional bits: 8-bit shifted coords * 8-bit shifted delta)
        int32_t float_simd_as_fixed = static_cast<int32_t>(std::round(hwy_float_results[i] * 65536.0f));
        int32_t fixed_simd_val = hwy_fixed_results[i];

        if (std::abs(float_simd_as_fixed - scalar_val) > 1) {
            float_discrepancies++;
        }
        if (fixed_simd_val != scalar_val) {
            fixed_discrepancies++;
        }
    }

    std::cout << "\n========================================================================\n";
    std::cout << "          PERFORMANCE METRICS & DETERMINISM REPORT (1M PIXELS)           \n";
    std::cout << "========================================================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "1. Scalar Reference (Fixed-Point): " << time_scalar_ms << " ms\n";
    std::cout << "2. Highway Float SIMD:             " << time_hwy_float_ms << " ms  (Speedup: " << (time_scalar_ms / time_hwy_float_ms) << "x)\n";
    std::cout << "3. Highway Fixed-Point SIMD:       " << time_hwy_fixed_ms << " ms  (Speedup: " << (time_scalar_ms / time_hwy_fixed_ms) << "x)\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "DETERMINISM VERIFICATION:\n";
    std::cout << " - HWY Float SIMD Discrepancies vs Scalar:       " << float_discrepancies << " / " << NUM_PIXELS << " pixels\n";
    std::cout << " - HWY Fixed-Point SIMD Discrepancies vs Scalar: " << fixed_discrepancies << " / " << NUM_PIXELS << " pixels\n";

    if (fixed_discrepancies == 0) {
        std::cout << "\nSTATUS: PASSED - Highway Fixed-Point SIMD achieves Bit-Exact Determinism!\n";
    } else {
        std::cout << "\nSTATUS: FAILED - Discrepancy found.\n";
    }
    std::cout << "========================================================================\n";

    return 0;
}
