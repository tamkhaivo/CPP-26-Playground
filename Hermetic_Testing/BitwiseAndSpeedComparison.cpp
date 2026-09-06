#include "HermeticMath.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>

using namespace Type0::Math;

struct TestPoint {
    const char* name;
    float x;
    float y;
};

void DetailedComparison() {
    std::cout << "========================================================================================================\n";
    std::cout << "             DETAILED BITWISE COMPARISON: std:: (OS Libm) VS Google Highway (Hermetic SIMD)             \n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left 
              << std::setw(8)  << "Func"
              << std::setw(12) << "Input X"
              << std::setw(12) << "std:: Float"
              << std::setw(12) << "std:: Hex"
              << std::setw(12) << "HWY Float"
              << std::setw(12) << "HWY Hex"
              << std::setw(10) << "ULP Diff"
              << std::setw(12) << "Bit Exact?" << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    auto CompareFunc = [](const char* name, float x, float y, float std_val, float hwy_val) {
        uint32_t u_std, u_hwy;
        std::memcpy(&u_std, &std_val, sizeof(float));
        std::memcpy(&u_hwy, &hwy_val, sizeof(float));

        int32_t ulp_diff = std::abs(static_cast<int32_t>(u_std) - static_cast<int32_t>(u_hwy));

        std::cout << std::left 
                  << std::setw(8)  << name
                  << std::setw(12) << std::setprecision(5) << x
                  << std::setw(12) << std::setprecision(6) << std_val
                  << "0x" << std::hex << std::setw(10) << u_std << std::dec
                  << std::setw(12) << std::setprecision(6) << hwy_val
                  << "0x" << std::hex << std::setw(10) << u_hwy << std::dec
                  << std::setw(10) << ulp_diff
                  << (ulp_diff == 0 ? "✅ MATCH" : "❌ DRIFT") << "\n";
    };

    std::vector<float> test_inputs = { 0.001f, 0.5f, 1.23456789f, 3.14159265f / 4.0f, 2.71828182f, 10.5f, 100.0f };

    std::cout << "--- SIN ---\n";
    for (float x : test_inputs) {
        CompareFunc("sin", x, 0.0f, std::sin(x), HermeticMath::Sin(x));
    }

    std::cout << "\n--- COS ---\n";
    for (float x : test_inputs) {
        CompareFunc("cos", x, 0.0f, std::cos(x), HermeticMath::Cos(x));
    }

    std::cout << "\n--- EXP ---\n";
    for (float x : test_inputs) {
        if (x > 20.0f) continue;
        CompareFunc("exp", x, 0.0f, std::exp(x), HermeticMath::Exp(x));
    }

    std::cout << "\n--- LOG ---\n";
    for (float x : test_inputs) {
        CompareFunc("log", x, 0.0f, std::log(x), HermeticMath::Log(x));
    }

    std::cout << "\n--- POW (x^1.5) ---\n";
    for (float x : test_inputs) {
        CompareFunc("pow", x, 1.5f, std::pow(x, 1.5f), HermeticMath::Pow(x, 1.5f));
    }

    std::cout << "\n--- ATAN2 (y=1.0, x) ---\n";
    for (float x : test_inputs) {
        CompareFunc("atan2", x, 1.0f, std::atan2(1.0f, x), HermeticMath::Atan2(1.0f, x));
    }
}

void BenchmarkSpeed() {
    std::cout << "\n========================================================================================================\n";
    std::cout << "                         BENCHMARK: std:: (Scalar Loop) VS Google Highway (SIMD)                       \n";
    std::cout << "========================================================================================================\n";

    constexpr size_t N = 10'000'000;
    std::vector<float> inputs(N);
    std::vector<float> outputs_std(N);
    std::vector<float> outputs_hwy(N);

    for (size_t i = 0; i < N; ++i) {
        inputs[i] = 0.0001f + static_cast<float>(i % 1000) * 0.001f;
    }

    // Benchmark std:: sin + cos loop
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        outputs_std[i] = std::sin(inputs[i]) + std::cos(inputs[i]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_std = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Benchmark Google Highway SIMD sin + cos loop
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i += lanes) {
        auto vx = hn::LoadU(d, inputs.data() + i);
        auto v_sin = hn::Sin(d, vx);
        auto v_cos = hn::Cos(d, vx);
        auto v_sum = hn::Add(v_sin, v_cos);
        hn::StoreU(v_sum, d, outputs_hwy.data() + i);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_hwy = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "std:: (Scalar C Runtime Loop 10M elements): " << std::fixed << std::setprecision(2) << ms_std << " ms\n";
    std::cout << "Google Highway (SIMD Loop 10M elements):   " << std::fixed << std::setprecision(2) << ms_hwy << " ms\n";
    std::cout << "🚀 SPEEDUP: " << std::setprecision(2) << (ms_std / ms_hwy) << "x FASTER!\n";
}

int main() {
    DetailedComparison();
    BenchmarkSpeed();
    return 0;
}
