#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>

// Include our emulation layer for std::simd
#include "std_simd_emulation.hpp"

// Include Google Highway
#include <hwy/highway.h>

namespace stdx = std::experimental;
namespace hn = hwy::HWY_NAMESPACE;

// Case 1 Implementation
void AddVectorsStd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t size) {
    size_t i = 0;
    constexpr size_t width = stdx::native_simd<float>::size();
    for (; i + width <= size; i += width) {
        stdx::native_simd<float> va, vb;
        va.copy_from(a + i, stdx::element_aligned);
        vb.copy_from(b + i, stdx::element_aligned);
        auto vc = va + vb;
        vc.copy_to(c + i, stdx::element_aligned);
    }
    for (; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

void AddVectorsHwy(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t size) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    size_t i = 0;
    for (; i + lanes <= size; i += lanes) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        auto vc = hn::Add(va, vb);
        hn::StoreU(vc, d, c + i);
    }
    for (; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Case 2 Implementation
void SaxpyStd(float a, const float* __restrict x, const float* __restrict y, float* __restrict z, size_t size) {
    size_t i = 0;
    constexpr size_t width = stdx::native_simd<float>::size();
    const stdx::native_simd<float> va(a);
    for (; i + width <= size; i += width) {
        stdx::native_simd<float> vx, vy;
        vx.copy_from(x + i, stdx::element_aligned);
        vy.copy_from(y + i, stdx::element_aligned);
        auto vz = va * vx + vy;
        vz.copy_to(z + i, stdx::element_aligned);
    }
    for (; i < size; ++i) {
        z[i] = a * x[i] + y[i];
    }
}

void SaxpyHwy(float a, const float* __restrict x, const float* __restrict y, float* __restrict z, size_t size) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const auto va = hn::Set(d, a);
    size_t i = 0;
    for (; i + lanes <= size; i += lanes) {
        auto vx = hn::LoadU(d, x + i);
        auto vy = hn::LoadU(d, y + i);
        auto vz = hn::MulAdd(va, vx, vy);
        hn::StoreU(vz, d, z + i);
    }
    for (; i < size; ++i) {
        z[i] = a * x[i] + y[i];
    }
}

// Case 3 Implementation
void ReluStd(const float* __restrict input, float* __restrict output, size_t size) {
    size_t i = 0;
    constexpr size_t width = stdx::native_simd<float>::size();
    const stdx::native_simd<float> zero(0.0f);
    for (; i + width <= size; i += width) {
        stdx::native_simd<float> vin;
        vin.copy_from(input + i, stdx::element_aligned);
        auto mask = vin > zero;
        stdx::native_simd<float> vout = zero;
        where(mask, vout) = vin;
        vout.copy_to(output + i, stdx::element_aligned);
    }
    for (; i < size; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

void ReluHwy(const float* __restrict input, float* __restrict output, size_t size) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    size_t i = 0;
    for (; i + lanes <= size; i += lanes) {
        auto vin = hn::LoadU(d, input + i);
        auto mask = hn::Gt(vin, zero);
        auto vout = hn::IfThenElse(mask, vin, zero);
        hn::StoreU(vout, d, output + i);
    }
    for (; i < size; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

// Case 4 Implementation
float DotProductStd(const float* __restrict a, const float* __restrict b, size_t size) {
    size_t i = 0;
    constexpr size_t width = stdx::native_simd<float>::size();
    stdx::native_simd<float> acc(0.0f);
    for (; i + width <= size; i += width) {
        stdx::native_simd<float> va, vb;
        va.copy_from(a + i, stdx::element_aligned);
        vb.copy_from(b + i, stdx::element_aligned);
        acc += va * vb;
    }
    float sum = stdx::reduce(acc);
    for (; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float DotProductHwy(const float* __restrict a, const float* __restrict b, size_t size) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    auto acc = hn::Zero(d);
    size_t i = 0;
    for (; i + lanes <= size; i += lanes) {
        auto va = hn::LoadU(d, a + i);
        auto vb = hn::LoadU(d, b + i);
        acc = hn::MulAdd(va, vb, acc);
    }
    float sum = hn::ReduceSum(d, acc);
    for (; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Case 5 Implementation
void ScalePoints3DStd(const float* __restrict xyz_in, float* __restrict xyz_out, float scale, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        xyz_out[i * 3 + 0] = xyz_in[i * 3 + 0] * scale;
        xyz_out[i * 3 + 1] = xyz_in[i * 3 + 1] * scale;
        xyz_out[i * 3 + 2] = xyz_in[i * 3 + 2] * scale;
    }
}

void ScalePoints3DHwy(const float* __restrict xyz_in, float* __restrict xyz_out, float scale, size_t count) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const auto v_scale = hn::Set(d, scale);
    size_t i = 0;
    for (; i + lanes <= count; i += lanes) {
        hn::Vec<decltype(d)> vx, vy, vz;
        hn::LoadInterleaved3(d, xyz_in + 3 * i, vx, vy, vz);
        vx = hn::Mul(vx, v_scale);
        vy = hn::Mul(vy, v_scale);
        vz = hn::Mul(vz, v_scale);
        hn::StoreInterleaved3(vx, vy, vz, d, xyz_out + 3 * i);
    }
    for (; i < count; ++i) {
        xyz_out[i * 3 + 0] = xyz_in[i * 3 + 0] * scale;
        xyz_out[i * 3 + 1] = xyz_in[i * 3 + 1] * scale;
        xyz_out[i * 3 + 2] = xyz_in[i * 3 + 2] * scale;
    }
}

// Case 6 Implementation
void AddVectorsMaskedStd(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t size) {
    size_t i = 0;
    constexpr size_t width = stdx::native_simd<float>::size();
    for (; i < size; i += width) {
        size_t remaining = size - i;
        if (remaining >= width) {
            stdx::native_simd<float> va, vb;
            va.copy_from(a + i, stdx::element_aligned);
            vb.copy_from(b + i, stdx::element_aligned);
            auto vc = va + vb;
            vc.copy_to(c + i, stdx::element_aligned);
        } else {
            stdx::native_simd_mask<float> mask([&](size_t idx) { return idx < remaining; });
            stdx::native_simd<float> va = 0.0f;
            stdx::native_simd<float> vb = 0.0f;
            va.copy_from(a + i, mask, stdx::element_aligned);
            vb.copy_from(b + i, mask, stdx::element_aligned);
            auto vc = va + vb;
            vc.copy_to(c + i, mask, stdx::element_aligned);
        }
    }
}

void AddVectorsMaskedHwy(const float* __restrict a, const float* __restrict b, float* __restrict c, size_t size) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    size_t i = 0;
    for (; i < size; i += lanes) {
        const size_t remaining = size - i;
        auto va = hn::LoadN(d, a + i, remaining);
        auto vb = hn::LoadN(d, b + i, remaining);
        auto vc = hn::Add(va, vb);
        hn::StoreN(vc, d, c + i, remaining);
    }
}

// Case 7 Implementation
void GatherScatterStd(const float* __restrict src, const int* __restrict indices, float* __restrict dest, size_t size) {
    size_t i = 0;
    constexpr size_t width = stdx::native_simd<float>::size();
    using index_simd = stdx::simd<int, stdx::simd_abi::native<float>>;
    for (; i + width <= size; i += width) {
        index_simd idx;
        idx.copy_from(indices + i, stdx::element_aligned);
        stdx::native_simd<float> val(src, idx);
        auto res = val * 2.0f;
        res.copy_to(dest, idx);
    }
    for (; i < size; ++i) {
        dest[indices[i]] = src[indices[i]] * 2.0f;
    }
}

void GatherScatterHwy(const float* __restrict src, const int* __restrict indices, float* __restrict dest, size_t size) {
    const hn::ScalableTag<float> df;
    const hn::ScalableTag<int> di;
    const size_t lanes = hn::Lanes(df);
    size_t i = 0;
    const auto two = hn::Set(df, 2.0f);
    for (; i + lanes <= size; i += lanes) {
        auto idx = hn::LoadU(di, indices + i);
        auto val = hn::GatherIndex(df, src, idx);
        auto res = hn::Mul(val, two);
        hn::ScatterIndex(res, df, dest, idx);
    }
    for (; i < size; ++i) {
        dest[indices[i]] = src[indices[i]] * 2.0f;
    }
}

// Benchmark Runner Helper
template<typename F>
double BenchmarkFunction(F func, int iterations) {
    // Warmup
    func();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    return elapsed.count() / iterations;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "     SIMD API Performance Benchmark: std::simd vs HWY   \n";
    std::cout << "========================================================\n\n";

    const size_t N = 100000;  // 100k elements
    const int iterations = 100;

    // Allocate aligned data
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N, 0.0f);

    std::vector<float> points_in(N * 3, 1.0f);
    std::vector<float> points_out(N * 3, 0.0f);

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    // Shuffle indices for gather/scatter
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::cout << "Running benchmarks with N = " << N << " elements over " << iterations << " iterations...\n\n";

    // Case 1
    double time_c1_std = BenchmarkFunction([&]() { AddVectorsStd(a.data(), b.data(), c.data(), N); }, iterations);
    double time_c1_hwy = BenchmarkFunction([&]() { AddVectorsHwy(a.data(), b.data(), c.data(), N); }, iterations);

    // Case 2
    double time_c2_std = BenchmarkFunction([&]() { SaxpyStd(2.5f, a.data(), b.data(), c.data(), N); }, iterations);
    double time_c2_hwy = BenchmarkFunction([&]() { SaxpyHwy(2.5f, a.data(), b.data(), c.data(), N); }, iterations);

    // Case 3
    double time_c3_std = BenchmarkFunction([&]() { ReluStd(a.data(), c.data(), N); }, iterations);
    double time_c3_hwy = BenchmarkFunction([&]() { ReluHwy(a.data(), c.data(), N); }, iterations);

    // Case 4
    float dot_std = 0.0f;
    float dot_hwy = 0.0f;
    double time_c4_std = BenchmarkFunction([&]() { dot_std = DotProductStd(a.data(), b.data(), N); }, iterations);
    double time_c4_hwy = BenchmarkFunction([&]() { dot_hwy = DotProductHwy(a.data(), b.data(), N); }, iterations);

    // Case 5
    double time_c5_std = BenchmarkFunction([&]() { ScalePoints3DStd(points_in.data(), points_out.data(), 1.5f, N); }, iterations);
    double time_c5_hwy = BenchmarkFunction([&]() { ScalePoints3DHwy(points_in.data(), points_out.data(), 1.5f, N); }, iterations);

    // Case 6
    // Make size non-multiple of SIMD width to exercise tail masking
    const size_t N_tail = N - 3;
    double time_c6_std = BenchmarkFunction([&]() { AddVectorsMaskedStd(a.data(), b.data(), c.data(), N_tail); }, iterations);
    double time_c6_hwy = BenchmarkFunction([&]() { AddVectorsMaskedHwy(a.data(), b.data(), c.data(), N_tail); }, iterations);

    // Case 7
    double time_c7_std = BenchmarkFunction([&]() { GatherScatterStd(a.data(), indices.data(), c.data(), N); }, iterations);
    double time_c7_hwy = BenchmarkFunction([&]() { GatherScatterHwy(a.data(), indices.data(), c.data(), N); }, iterations);

    // Print Results
    std::cout << "====================================================================\n";
    std::cout << "                    COMPARATIVE PERFORMANCE TABLE                    \n";
    std::cout << "====================================================================\n";
    std::cout << std::left << std::setw(28) << "Complexity Case" 
              << " | " << std::right << std::setw(15) << "std::simd (μs)" 
              << " | " << std::right << std::setw(15) << "Hwy SIMD (μs)" 
              << " | Speedup Ratio\n";
    std::cout << "--------------------------------------------------------------------\n";
    
    auto print_row = [](const std::string& name, double t_std, double t_hwy) {
        double ratio = t_std / t_hwy;
        std::cout << std::left << std::setw(28) << name 
                  << " | " << std::right << std::setw(12) << std::fixed << std::setprecision(2) << t_std << " μs"
                  << " | " << std::right << std::setw(12) << t_hwy << " μs"
                  << " | " << std::right << std::setw(10) << std::setprecision(2) << ratio << "x\n";
    };

    print_row("Case 1: Vector Addition", time_c1_std, time_c1_hwy);
    print_row("Case 2: Saxpy (A*X + Y)", time_c2_std, time_c2_hwy);
    print_row("Case 3: Conditional Mask", time_c3_std, time_c3_hwy);
    print_row("Case 4: Dot Product (Red.)", time_c4_std, time_c4_hwy);
    print_row("Case 5: AoS to SoA (Load3)", time_c5_std, time_c5_hwy);
    print_row("Case 6: Masked Tail End", time_c6_std, time_c6_hwy);
    print_row("Case 7: Gather & Scatter", time_c7_std, time_c7_hwy);
    
    std::cout << "====================================================================\n\n";

    return 0;
}
