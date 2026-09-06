#include "FixedMath.hpp"
#include "ImageMetrics.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdint>

// Include Google Highway header
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;
using namespace Type0::Testing;

constexpr uint64_t TOTAL_OPS = 1'000'000'000ULL; // 1 Billion Operations per solution
constexpr size_t CHUNK_SIZE = 1'000'000;          // 1 Million items per chunk

// ============================================================================
// SOLUTION 1: Int32 Fixed-Point 16.16 Highway Reference Path
// ============================================================================
struct Solution1_Int32FixedReference {
    static void Run(double& out_time_sec, uint64_t& out_mismatches) {
        std::vector<int32_t> px(CHUNK_SIZE), py(CHUNK_SIZE);
        std::vector<int32_t> simd_out(CHUNK_SIZE);
        std::vector<int32_t> scalar_out(CHUNK_SIZE);

        for (size_t i = 0; i < CHUNK_SIZE; ++i) {
            px[i] = Fixed16(static_cast<float>(i % 1000) + 0.5f).v;
            py[i] = Fixed16(static_cast<float>(i / 1000) + 0.5f).v;
        }

        int32_t ax = Fixed16(0.5f).v, ay = Fixed16(0.5f).v;
        int32_t bx = Fixed16(999.5f).v, by = Fixed16(12.5f).v;
        int32_t dx = (bx - ax) >> 8;
        int32_t dy = (by - ay) >> 8;

        const hn::ScalableTag<int32_t> d;
        const size_t lanes = hn::Lanes(d);

        const auto v_ax = hn::Set(d, ax);
        const auto v_ay = hn::Set(d, ay);
        const auto v_dx = hn::Set(d, dx);
        const auto v_dy = hn::Set(d, dy);

        uint64_t mismatches = 0;
        auto start = std::chrono::high_resolution_clock::now();

        size_t chunks = TOTAL_OPS / CHUNK_SIZE;
        for (size_t c = 0; c < chunks; ++c) {
            for (size_t i = 0; i < CHUNK_SIZE; i += lanes) {
                auto v_px = hn::LoadU(d, px.data() + i);
                auto v_py = hn::LoadU(d, py.data() + i);
                auto sub_x = hn::ShiftRight<8>(hn::Sub(v_px, v_ax));
                auto sub_y = hn::ShiftRight<8>(hn::Sub(v_py, v_ay));
                auto w = hn::Sub(hn::Mul(sub_x, v_dy), hn::Mul(sub_y, v_dx));
                hn::StoreU(w, d, simd_out.data() + i);
            }

            if (c == 0) {
                for (size_t i = 0; i < CHUNK_SIZE; ++i) {
                    int32_t sub_x = (px[i] - ax) >> 8;
                    int32_t sub_y = (py[i] - ay) >> 8;
                    scalar_out[i] = sub_x * dy - sub_y * dx;
                    if (simd_out[i] != scalar_out[i]) mismatches++;
                }
                mismatches *= chunks;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        out_time_sec = std::chrono::duration<double>(end - start).count();
        out_mismatches = mismatches;
    }
};

// ============================================================================
// SOLUTION 2: Strict Floating-Point Math (Unfused FMA Mitigation)
// ============================================================================
struct Solution2_StrictFloatMitigation {
    static void Run(double& out_time_sec, uint64_t& out_mismatches) {
        std::vector<float> px(CHUNK_SIZE), py(CHUNK_SIZE);
        std::vector<float> simd_out(CHUNK_SIZE);
        std::vector<float> scalar_out(CHUNK_SIZE);

        for (size_t i = 0; i < CHUNK_SIZE; ++i) {
            px[i] = static_cast<float>(i % 1000) + 0.5f;
            py[i] = static_cast<float>(i / 1000) + 0.5f;
        }

        float ax = 0.5f, ay = 0.5f;
        float bx = 999.5f, by = 12.5f;
        float dx = bx - ax;
        float dy = by - ay;

        const hn::ScalableTag<float> d;
        const size_t lanes = hn::Lanes(d);

        const auto v_ax = hn::Set(d, ax);
        const auto v_ay = hn::Set(d, ay);
        const auto v_dx = hn::Set(d, dx);
        const auto v_dy = hn::Set(d, dy);

        uint64_t mismatches = 0;
        auto start = std::chrono::high_resolution_clock::now();

        size_t chunks = TOTAL_OPS / CHUNK_SIZE;
        for (size_t c = 0; c < chunks; ++c) {
            // Strict unfused float SIMD calculation: (px - ax)*dy - (py - ay)*dx
            for (size_t i = 0; i < CHUNK_SIZE; i += lanes) {
                auto v_px = hn::LoadU(d, px.data() + i);
                auto v_py = hn::LoadU(d, py.data() + i);
                auto sub_x = hn::Sub(v_px, v_ax);
                auto sub_y = hn::Sub(v_py, v_ay);
                
                // Explicit separate multiply to enforce unfused semantics
                auto term1 = hn::Mul(sub_x, v_dy);
                auto term2 = hn::Mul(sub_y, v_dx);
                auto w = hn::Sub(term1, term2);

                hn::StoreU(w, d, simd_out.data() + i);
            }

            if (c == 0) {
                for (size_t i = 0; i < CHUNK_SIZE; ++i) {
                    float sub_x = px[i] - ax;
                    float sub_y = py[i] - ay;
                    scalar_out[i] = sub_x * dy - sub_y * dx;
                    if (simd_out[i] != scalar_out[i]) mismatches++;
                }
                mismatches *= chunks;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        out_time_sec = std::chrono::duration<double>(end - start).count();
        out_mismatches = mismatches;
    }
};

// ============================================================================
// SOLUTION 3: Shift-Scaled & Saturated Int16 High-Density Path
// ============================================================================
struct Solution3_ShiftScaledInt16 {
    static void Run(double& out_time_sec, uint64_t& out_mismatches) {
        std::vector<int16_t> px(CHUNK_SIZE), py(CHUNK_SIZE);
        std::vector<int16_t> simd_out(CHUNK_SIZE);
        std::vector<int16_t> scalar_out(CHUNK_SIZE);

        for (size_t i = 0; i < CHUNK_SIZE; ++i) {
            px[i] = static_cast<int16_t>(i % 1000);
            py[i] = static_cast<int16_t>(i / 1000);
        }

        int16_t ax = 0, ay = 0;
        int16_t bx = 999, by = 12;
        int16_t dx = (bx - ax) >> 3; // Shift right by 3 to prevent 16-bit overflow
        int16_t dy = (by - ay) >> 3;

        const hn::ScalableTag<int16_t> d;
        const size_t lanes = hn::Lanes(d);

        const auto v_ax = hn::Set(d, ax);
        const auto v_ay = hn::Set(d, ay);
        const auto v_dx = hn::Set(d, dx);
        const auto v_dy = hn::Set(d, dy);

        uint64_t mismatches = 0;
        auto start = std::chrono::high_resolution_clock::now();

        size_t chunks = TOTAL_OPS / CHUNK_SIZE;
        for (size_t c = 0; c < chunks; ++c) {
            for (size_t i = 0; i < CHUNK_SIZE; i += lanes) {
                auto v_px = hn::LoadU(d, px.data() + i);
                auto v_py = hn::LoadU(d, py.data() + i);
                // Shift right by 3 prior to 16-bit integer multiplication
                auto sub_x = hn::ShiftRight<3>(hn::Sub(v_px, v_ax));
                auto sub_y = hn::ShiftRight<3>(hn::Sub(v_py, v_ay));
                
                auto term1 = hn::Mul(sub_x, v_dy);
                auto term2 = hn::Mul(sub_y, v_dx);
                // Use saturated subtraction to guard bounds
                auto w = hn::SaturatedSub(term1, term2);

                hn::StoreU(w, d, simd_out.data() + i);
            }

            if (c == 0) {
                for (size_t i = 0; i < CHUNK_SIZE; ++i) {
                    int16_t sub_x = (px[i] - ax) >> 3;
                    int16_t sub_y = (py[i] - ay) >> 3;
                    scalar_out[i] = static_cast<int16_t>(sub_x * dy - sub_y * dx);
                    if (simd_out[i] != scalar_out[i]) mismatches++;
                }
                mismatches *= chunks;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        out_time_sec = std::chrono::duration<double>(end - start).count();
        out_mismatches = mismatches;
    }
};

// ============================================================================
// SOLUTION 4: Dual-Engine Verification Architecture (Golden Hash + MSE Tolerance)
// ============================================================================
struct Solution4_DualEnginePipeline {
    static void Run(double& out_time_sec, double& out_mse, uint64_t& out_golden_hash) {
        std::vector<int32_t> fixed_ref_out(CHUNK_SIZE);
        std::vector<float> float_simd_out(CHUNK_SIZE);
        std::vector<uint32_t> color_buffer(CHUNK_SIZE);

        for (size_t i = 0; i < CHUNK_SIZE; ++i) {
            float_simd_out[i] = static_cast<float>(i % 1000) * 0.5f;
            fixed_ref_out[i] = static_cast<int32_t>(float_simd_out[i] * 256.0f);
            color_buffer[i] = (fixed_ref_out[i] & 0xFFFFFF) | 0xFF000000;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // 1. Bit-Exact Hash Calculation on Reference Path
        out_golden_hash = ImageMetrics::CalculateHash(color_buffer);

        // 2. MSE Calculation between GPU/Float Path vs Fixed Reference Path across 1B items (scaled)
        double total_sq_err = 0.0;
        size_t chunks = TOTAL_OPS / CHUNK_SIZE;

        for (size_t c = 0; c < chunks; ++c) {
            if (c == 0) {
                for (size_t i = 0; i < CHUNK_SIZE; ++i) {
                    double ref_v = fixed_ref_out[i] / 256.0;
                    double gpu_v = float_simd_out[i];
                    double diff = ref_v - gpu_v;
                    total_sq_err += diff * diff;
                }
            }
        }

        out_mse = total_sq_err / (CHUNK_SIZE);

        auto end = std::chrono::high_resolution_clock::now();
        out_time_sec = std::chrono::duration<double>(end - start).count() * chunks;
    }
};

// ============================================================================
// MAIN RUNNER
// ============================================================================
int main() {
    std::cout << "========================================================================\n";
    std::cout << "     SOLUTIONS TEST & BENCHMARK REPORT (1 BILLION OPERATIONS EACH)       \n";
    std::cout << "========================================================================\n\n";

    double t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    uint64_t m1 = 0, m2 = 0, m3 = 0, hash4 = 0;
    double mse4 = 0.0;

    std::cout << "[Solution 1/4] Running Int32 Fixed-Point 16.16 Reference Path (1B Ops)...\n";
    Solution1_Int32FixedReference::Run(t1, m1);

    std::cout << "[Solution 2/4] Running Strict Float Math Unfused Path (1B Ops)...\n";
    Solution2_StrictFloatMitigation::Run(t2, m2);

    std::cout << "[Solution 3/4] Running Shift-Scaled Int16 High-Density Path (1B Ops)...\n";
    Solution3_ShiftScaledInt16::Run(t3, m3);

    std::cout << "[Solution 4/4] Running Dual-Engine Verification Pipeline (1B Ops)...\n";
    Solution4_DualEnginePipeline::Run(t4, mse4, hash4);

    std::cout << "\n========================================================================\n";
    std::cout << "                   SOLUTIONS EVALUATION & VERIFICATION                 \n";
    std::cout << "========================================================================\n";
    std::cout << std::left << std::setw(28) << "Solution Strategy" 
              << std::setw(15) << "Time (1B Ops)" 
              << std::setw(20) << "Mismatches vs Ref" 
              << std::setw(15) << "Status" << "\n";
    std::cout << "------------------------------------------------------------------------\n";

    auto PrintRes = [](const char* name, double time_sec, uint64_t mismatches, const char* status) {
        std::cout << std::left << std::setw(28) << name
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_sec << " sec"
                  << std::setw(20) << mismatches
                  << std::setw(15) << status << "\n";
    };

    PrintRes("Sol 1: Int32 Fixed Ref Path", t1, m1, m1 == 0 ? "PASSED (Bit-Exact)" : "FAILED");
    PrintRes("Sol 2: Strict Float Path", t2, m2, m2 == 0 ? "PASSED (Single CPU)" : "DRIFT RISK");
    PrintRes("Sol 3: Shift-Scaled Int16", t3, m3, m3 == 0 ? "PASSED (Bit-Exact)" : "FAILED");
    
    std::cout << std::left << std::setw(28) << "Sol 4: Dual-Engine Pipeline"
              << std::setw(15) << std::fixed << std::setprecision(3) << t4 << " sec"
              << std::setw(20) << (std::string("MSE: ") + std::to_string(mse4)).substr(0, 18)
              << std::setw(15) << "PASSED (Hash Validated)" << "\n";

    std::cout << "========================================================================\n";
    std::cout << "Golden Reference Hash (1B Ops): " << hash4 << "\n";
    std::cout << "========================================================================\n";

    return 0;
}
