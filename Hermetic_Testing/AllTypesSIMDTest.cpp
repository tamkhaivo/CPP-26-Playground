#include "FixedMath.hpp"
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

constexpr uint64_t TOTAL_OPERATIONS = 1'000'000'000ULL; // 1 Billion Operations
constexpr size_t CHUNK_SIZE = 1'000'000;                // 1 Million per chunk

// ============================================================================
// 1. FLOAT 32 BENCHMARK & DISCREPANCY TEST
// ============================================================================
struct Float32Test {
    static void Run(uint64_t total_ops, double& out_time_ms, uint64_t& out_mismatches) {
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

        size_t chunks = total_ops / CHUNK_SIZE;
        for (size_t c = 0; c < chunks; ++c) {
            // SIMD Pass
            for (size_t i = 0; i < CHUNK_SIZE; i += lanes) {
                auto v_px = hn::LoadU(d, px.data() + i);
                auto v_py = hn::LoadU(d, py.data() + i);
                auto sub_x = hn::Sub(v_px, v_ax);
                auto sub_y = hn::Sub(v_py, v_ay);
                auto w = hn::Sub(hn::Mul(sub_x, v_dy), hn::Mul(sub_y, v_dx));
                hn::StoreU(w, d, simd_out.data() + i);
            }

            // Scalar Pass for first chunk discrepancy check
            if (c == 0) {
                for (size_t i = 0; i < CHUNK_SIZE; ++i) {
                    float sub_x = px[i] - ax;
                    float sub_y = py[i] - ay;
                    scalar_out[i] = sub_x * dy - sub_y * dx;
                    if (simd_out[i] != scalar_out[i]) {
                        mismatches++;
                    }
                }
                // Extrapolate mismatch rate across 1B
                mismatches = mismatches * chunks;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        out_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        out_mismatches = mismatches;
    }
};

// ============================================================================
// 2. DOUBLE 64 BENCHMARK & DISCREPANCY TEST
// ============================================================================
struct Double64Test {
    static void Run(uint64_t total_ops, double& out_time_ms, uint64_t& out_mismatches) {
        std::vector<double> px(CHUNK_SIZE), py(CHUNK_SIZE);
        std::vector<double> simd_out(CHUNK_SIZE);
        std::vector<double> scalar_out(CHUNK_SIZE);

        for (size_t i = 0; i < CHUNK_SIZE; ++i) {
            px[i] = static_cast<double>(i % 1000) + 0.5;
            py[i] = static_cast<double>(i / 1000) + 0.5;
        }

        double ax = 0.5, ay = 0.5;
        double bx = 999.5, by = 12.5;
        double dx = bx - ax;
        double dy = by - ay;

        const hn::ScalableTag<double> d;
        const size_t lanes = hn::Lanes(d);

        const auto v_ax = hn::Set(d, ax);
        const auto v_ay = hn::Set(d, ay);
        const auto v_dx = hn::Set(d, dx);
        const auto v_dy = hn::Set(d, dy);

        uint64_t mismatches = 0;
        auto start = std::chrono::high_resolution_clock::now();

        size_t chunks = total_ops / CHUNK_SIZE;
        for (size_t c = 0; c < chunks; ++c) {
            // SIMD Pass
            for (size_t i = 0; i < CHUNK_SIZE; i += lanes) {
                auto v_px = hn::LoadU(d, px.data() + i);
                auto v_py = hn::LoadU(d, py.data() + i);
                auto sub_x = hn::Sub(v_px, v_ax);
                auto sub_y = hn::Sub(v_py, v_ay);
                auto w = hn::Sub(hn::Mul(sub_x, v_dy), hn::Mul(sub_y, v_dx));
                hn::StoreU(w, d, simd_out.data() + i);
            }

            if (c == 0) {
                for (size_t i = 0; i < CHUNK_SIZE; ++i) {
                    double sub_x = px[i] - ax;
                    double sub_y = py[i] - ay;
                    scalar_out[i] = sub_x * dy - sub_y * dx;
                    if (simd_out[i] != scalar_out[i]) {
                        mismatches++;
                    }
                }
                mismatches = mismatches * chunks;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        out_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        out_mismatches = mismatches;
    }
};

// ============================================================================
// 3. INT16 FIXED-POINT BENCHMARK & DISCREPANCY TEST
// ============================================================================
struct Int16Test {
    static void Run(uint64_t total_ops, double& out_time_ms, uint64_t& out_mismatches) {
        std::vector<int16_t> px(CHUNK_SIZE), py(CHUNK_SIZE);
        std::vector<int16_t> simd_out(CHUNK_SIZE);
        std::vector<int16_t> scalar_out(CHUNK_SIZE);

        for (size_t i = 0; i < CHUNK_SIZE; ++i) {
            px[i] = static_cast<int16_t>((i % 1000));
            py[i] = static_cast<int16_t>((i / 1000));
        }

        int16_t ax = 0, ay = 0;
        int16_t bx = 999, by = 12;
        int16_t dx = bx - ax;
        int16_t dy = by - ay;

        const hn::ScalableTag<int16_t> d;
        const size_t lanes = hn::Lanes(d);

        const auto v_ax = hn::Set(d, ax);
        const auto v_ay = hn::Set(d, ay);
        const auto v_dx = hn::Set(d, dx);
        const auto v_dy = hn::Set(d, dy);

        uint64_t mismatches = 0;
        auto start = std::chrono::high_resolution_clock::now();

        size_t chunks = total_ops / CHUNK_SIZE;
        for (size_t c = 0; c < chunks; ++c) {
            for (size_t i = 0; i < CHUNK_SIZE; i += lanes) {
                auto v_px = hn::LoadU(d, px.data() + i);
                auto v_py = hn::LoadU(d, py.data() + i);
                auto sub_x = hn::Sub(v_px, v_ax);
                auto sub_y = hn::Sub(v_py, v_ay);
                auto w = hn::Sub(hn::Mul(sub_x, v_dy), hn::Mul(sub_y, v_dx));
                hn::StoreU(w, d, simd_out.data() + i);
            }

            if (c == 0) {
                for (size_t i = 0; i < CHUNK_SIZE; ++i) {
                    int16_t sub_x = px[i] - ax;
                    int16_t sub_y = py[i] - ay;
                    scalar_out[i] = static_cast<int16_t>(sub_x * dy - sub_y * dx);
                    if (simd_out[i] != scalar_out[i]) {
                        mismatches++;
                    }
                }
                mismatches = mismatches * chunks;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        out_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        out_mismatches = mismatches;
    }
};

// ============================================================================
// 4. INT32 FIXED-POINT BENCHMARK & DISCREPANCY TEST
// ============================================================================
struct Int32Test {
    static void Run(uint64_t total_ops, double& out_time_ms, uint64_t& out_mismatches) {
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

        size_t chunks = total_ops / CHUNK_SIZE;
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
                    if (simd_out[i] != scalar_out[i]) {
                        mismatches++;
                    }
                }
                mismatches = mismatches * chunks;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        out_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        out_mismatches = mismatches;
    }
};

// ============================================================================
// 5. INT64 FIXED-POINT BENCHMARK & DISCREPANCY TEST
// ============================================================================
struct Int64Test {
    static void Run(uint64_t total_ops, double& out_time_ms, uint64_t& out_mismatches) {
        std::vector<int64_t> px(CHUNK_SIZE), py(CHUNK_SIZE);
        std::vector<int64_t> simd_out(CHUNK_SIZE);
        std::vector<int64_t> scalar_out(CHUNK_SIZE);

        for (size_t i = 0; i < CHUNK_SIZE; ++i) {
            px[i] = static_cast<int64_t>((i % 1000)) << 16;
            py[i] = static_cast<int64_t>((i / 1000)) << 16;
        }

        int64_t ax = 0, ay = 0;
        int64_t bx = 999LL << 16, by = 12LL << 16;
        int64_t dx = (bx - ax) >> 16;
        int64_t dy = (by - ay) >> 16;

        const hn::ScalableTag<int64_t> d;
        const size_t lanes = hn::Lanes(d);

        const auto v_ax = hn::Set(d, ax);
        const auto v_ay = hn::Set(d, ay);
        const auto v_dx = hn::Set(d, dx);
        const auto v_dy = hn::Set(d, dy);

        uint64_t mismatches = 0;
        auto start = std::chrono::high_resolution_clock::now();

        size_t chunks = total_ops / CHUNK_SIZE;
        for (size_t c = 0; c < chunks; ++c) {
            for (size_t i = 0; i < CHUNK_SIZE; i += lanes) {
                auto v_px = hn::LoadU(d, px.data() + i);
                auto v_py = hn::LoadU(d, py.data() + i);
                auto sub_x = hn::ShiftRight<16>(hn::Sub(v_px, v_ax));
                auto sub_y = hn::ShiftRight<16>(hn::Sub(v_py, v_ay));
                auto w = hn::Sub(hn::Mul(sub_x, v_dy), hn::Mul(sub_y, v_dx));
                hn::StoreU(w, d, simd_out.data() + i);
            }

            if (c == 0) {
                for (size_t i = 0; i < CHUNK_SIZE; ++i) {
                    int64_t sub_x = (px[i] - ax) >> 16;
                    int64_t sub_y = (py[i] - ay) >> 16;
                    scalar_out[i] = sub_x * dy - sub_y * dx;
                    if (simd_out[i] != scalar_out[i]) {
                        mismatches++;
                    }
                }
                mismatches = mismatches * chunks;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        out_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        out_mismatches = mismatches;
    }
};

// ============================================================================
// MAIN SUITE RUNNER
// ============================================================================
int main() {
    std::cout << "========================================================================\n";
    std::cout << "      COMPREHENSIVE ALL-DATA-TYPES BENCHMARK & DISCREPANCY REPORT       \n";
    std::cout << "                 TOTAL OPERATIONS: 1,000,000,000 (1 BILLION)            \n";
    std::cout << "========================================================================\n\n";

    double t_float32 = 0, t_double64 = 0, t_int16 = 0, t_int32 = 0, t_int64 = 0;
    uint64_t m_float32 = 0, m_double64 = 0, m_int16 = 0, m_int32 = 0, m_int64 = 0;

    std::cout << "Running 1B Operations Test for Float32...\n";
    Float32Test::Run(TOTAL_OPERATIONS, t_float32, m_float32);

    std::cout << "Running 1B Operations Test for Double64...\n";
    Double64Test::Run(TOTAL_OPERATIONS, t_double64, m_double64);

    std::cout << "Running 1B Operations Test for Int16...\n";
    Int16Test::Run(TOTAL_OPERATIONS, t_int16, m_int16);

    std::cout << "Running 1B Operations Test for Int32...\n";
    Int32Test::Run(TOTAL_OPERATIONS, t_int32, m_int32);

    std::cout << "Running 1B Operations Test for Int64...\n";
    Int64Test::Run(TOTAL_OPERATIONS, t_int64, m_int64);

    std::cout << "\n========================================================================\n";
    std::cout << "                        BENCHMARK RESULTS SUMMARY                       \n";
    std::cout << "========================================================================\n";
    std::cout << std::left << std::setw(15) << "Data Type" 
              << std::setw(18) << "Time (1B Ops)" 
              << std::setw(22) << "Mismatches vs Scalar" 
              << std::setw(15) << "Determinism" << "\n";
    std::cout << "------------------------------------------------------------------------\n";

    auto PrintRow = [](const char* name, double time_ms, uint64_t mismatches) {
        std::cout << std::left << std::setw(15) << name
                  << std::setw(12) << std::fixed << std::setprecision(2) << (time_ms / 1000.0) << " sec  "
                  << std::setw(22) << mismatches
                  << std::setw(15) << (mismatches == 0 ? "PASSED (Bit-Exact)" : "FAILED (Drift)") << "\n";
    };

    PrintRow("Float32", t_float32, m_float32);
    PrintRow("Double64", t_double64, m_double64);
    PrintRow("Int16 Fixed", t_int16, m_int16);
    PrintRow("Int32 Fixed", t_int32, m_int32);
    PrintRow("Int64 Fixed", t_int64, m_int64);

    std::cout << "========================================================================\n";

    return 0;
}
