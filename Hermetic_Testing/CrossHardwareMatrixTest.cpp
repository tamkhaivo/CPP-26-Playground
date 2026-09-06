#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>

#include "HermeticMath.hpp"
#include "ImageMetrics.hpp"

using namespace Type0::Math;
using namespace Type0::Testing;

struct MatrixTestResult {
    std::string name;
    uint32_t sin_hex;
    uint32_t cos_hex;
    uint32_t exp_hex;
    uint32_t log_hex;
    uint32_t pow_hex;
    uint32_t atan2_hex;
    uint64_t hash;
};

// 1. Standard C Runtime
MatrixTestResult TestStdLib() {
    MatrixTestResult r;
    r.name = "std:: (OS C Runtime)";

    float x = 1.23456789f;
    float y = 2.71828182f;

    float s = std::sin(x);
    float c = std::cos(x);
    float e = std::exp(x);
    float l = std::log(x);
    float p = std::pow(x, y);
    float a = std::atan2(y, x);

    std::memcpy(&r.sin_hex, &s, sizeof(float));
    std::memcpy(&r.cos_hex, &c, sizeof(float));
    std::memcpy(&r.exp_hex, &e, sizeof(float));
    std::memcpy(&r.log_hex, &l, sizeof(float));
    std::memcpy(&r.pow_hex, &p, sizeof(float));
    std::memcpy(&r.atan2_hex, &a, sizeof(float));

    constexpr size_t N = 100'000;
    std::vector<uint32_t> output(N);
    for (size_t i = 0; i < N; ++i) {
        float v = static_cast<float>(i) * 0.0001f + 0.00001f;
        float val = (std::sin(v) + std::cos(v)) + (std::exp(v) + std::log(v));
        std::memcpy(&output[i], &val, sizeof(float));
    }

    r.hash = ImageMetrics::CalculateHash(output);
    return r;
}

// 2. Google Highway SIMD CPU Vector Target
MatrixTestResult TestHighwaySIMD() {
    MatrixTestResult r;
    r.name = "Highway SIMD CPU";

    float x = 1.23456789f;
    float y = 2.71828182f;

    float s = HermeticMath::Sin(x);
    float c = HermeticMath::Cos(x);
    float e = HermeticMath::Exp(x);
    float l = HermeticMath::Log(x);
    float p = HermeticMath::Pow(x, y);
    float a = HermeticMath::Atan2(y, x);

    std::memcpy(&r.sin_hex, &s, sizeof(float));
    std::memcpy(&r.cos_hex, &c, sizeof(float));
    std::memcpy(&r.exp_hex, &e, sizeof(float));
    std::memcpy(&r.log_hex, &l, sizeof(float));
    std::memcpy(&r.pow_hex, &p, sizeof(float));
    std::memcpy(&r.atan2_hex, &a, sizeof(float));

    constexpr size_t N = 100'000;
    std::vector<uint32_t> output(N);

    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    std::vector<float> inputs(N);
    for (size_t i = 0; i < N; ++i) {
        inputs[i] = static_cast<float>(i) * 0.0001f + 0.00001f;
    }

    for (size_t i = 0; i < N; i += lanes) {
        auto v = hn::LoadU(d, inputs.data() + i);
        auto sin_v = HermeticMath::Sin(d, v);
        auto cos_v = HermeticMath::Cos(d, v);
        auto exp_v = HermeticMath::Exp(d, v);
        auto log_v = HermeticMath::Log(d, v);

        auto sum = hn::Add(hn::Add(sin_v, cos_v), hn::Add(exp_v, log_v));
        hn::StoreU(sum, d, reinterpret_cast<float*>(output.data() + i));
    }

    r.hash = ImageMetrics::CalculateHash(output);
    return r;
}

// 3. GPU Shader Simulation (Strict Hermetic Minimax Polynomial match)
MatrixTestResult TestGPUPolynomialShader() {
    MatrixTestResult r;
    r.name = "GPU Shader Polynomial";

    float x = 1.23456789f;
    float y = 2.71828182f;

    float s = HermeticMath::Sin(x);
    float c = HermeticMath::Cos(x);
    float e = HermeticMath::Exp(x);
    float l = HermeticMath::Log(x);
    float p = HermeticMath::Pow(x, y);
    float a = HermeticMath::Atan2(y, x);

    std::memcpy(&r.sin_hex, &s, sizeof(float));
    std::memcpy(&r.cos_hex, &c, sizeof(float));
    std::memcpy(&r.exp_hex, &e, sizeof(float));
    std::memcpy(&r.log_hex, &l, sizeof(float));
    std::memcpy(&r.pow_hex, &p, sizeof(float));
    std::memcpy(&r.atan2_hex, &a, sizeof(float));

    constexpr size_t N = 100'000;
    std::vector<uint32_t> output(N);

    for (size_t i = 0; i < N; ++i) {
        float v = static_cast<float>(i) * 0.0001f + 0.00001f;
        float sin_v = HermeticMath::Sin(v);
        float cos_v = HermeticMath::Cos(v);
        float exp_v = HermeticMath::Exp(v);
        float log_v = HermeticMath::Log(v);

        float sum = (sin_v + cos_v) + (exp_v + log_v);
        std::memcpy(&output[i], &sum, sizeof(float));
    }

    r.hash = ImageMetrics::CalculateHash(output);
    return r;
}

int main() {
    std::cout << "========================================================================================================\n";
    std::cout << "        MULTI-TARGET HARDWARE MATRIX TEST: std:: vs HIGHWAY CPU SIMD vs GPU SIMULATION                  \n";
    std::cout << "========================================================================================================\n";

    MatrixTestResult res_std = TestStdLib();
    MatrixTestResult res_hwy = TestHighwaySIMD();
    MatrixTestResult res_gpu = TestGPUPolynomialShader();

    std::cout << std::left 
              << std::setw(25) << "Execution Target"
              << std::setw(12) << "sin Hex"
              << std::setw(12) << "cos Hex"
              << std::setw(12) << "exp Hex"
              << std::setw(12) << "log Hex"
              << std::setw(12) << "pow Hex"
              << std::setw(12) << "atan2 Hex"
              << std::setw(20) << "100K Buffer Hash" << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    auto PrintRow = [](const MatrixTestResult& r) {
        std::cout << std::left << std::setw(25) << r.name
                  << "0x" << std::hex << std::setw(10) << r.sin_hex
                  << "0x" << std::hex << std::setw(10) << r.cos_hex
                  << "0x" << std::hex << std::setw(10) << r.exp_hex
                  << "0x" << std::hex << std::setw(10) << r.log_hex
                  << "0x" << std::hex << std::setw(10) << r.pow_hex
                  << "0x" << std::hex << std::setw(10) << r.atan2_hex
                  << "0x" << std::hex << std::setw(18) << r.hash << std::dec << "\n";
    };

    PrintRow(res_std);
    PrintRow(res_hwy);
    PrintRow(res_gpu);

    std::cout << "\n========================================================================================================\n";
    std::cout << "                                    BITWISE VERDICT & ANALYSIS                                          \n";
    std::cout << "========================================================================================================\n";

    if (res_hwy.hash == res_gpu.hash) {
        std::cout << "✅ CPU Highway SIMD & GPU Polynomial Engine: 100% BIT-IDENTICAL MATCH! (Hash: 0x" 
                  << std::hex << res_gpu.hash << std::dec << ")\n";
    } else {
        std::cout << "❌ CPU & GPU divergence detected!\n";
    }

    if (res_hwy.hash != res_std.hash) {
        std::cout << "⚠️ Standard C Runtime (`std::`) Hash: 0x" << std::hex << res_std.hash << std::dec << "\n";
        std::cout << "   -> `std::` DRIFTS from Highway & GPU! Standard libm calls OS-specific CRT libraries,\n";
        std::cout << "      breaking bitwise determinism across platform targets.\n";
    }

    std::cout << "========================================================================================================\n";

    return 0;
}
