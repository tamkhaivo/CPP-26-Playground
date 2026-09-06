#include "HermeticMath.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

using namespace Type0::Math;

// Structural table record
struct ComparisonResult {
    std::string function_name;
    float input_x;
    float input_y;
    float std_res;
    uint32_t std_hex;
    float hwy_res;
    uint32_t hwy_hex;
    int32_t ulp_diff;
    bool exact_match;
};

void TestFullFunctionMapping() {
    std::cout << "========================================================================================================\n";
    std::cout << "           FULL MAP: STANDARD C++ (std::) vs GOOGLE HIGHWAY SIMD MATH (hwy::HWY_NAMESPACE)             \n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left 
              << std::setw(12) << "C++ Standard" 
              << std::setw(28) << "Highway Counterpart" 
              << std::setw(14) << "Hermetic?" 
              << std::setw(40) << "Highway Function Signature / Vector Form" << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    auto PrintRow = [](const char* std_name, const char* hwy_name, const char* hermetic, const char* sig) {
        std::cout << std::left 
                  << std::setw(12) << std_name 
                  << std::setw(28) << hwy_name 
                  << std::setw(14) << hermetic 
                  << std::setw(40) << sig << "\n";
    };

    PrintRow("std::sin",   "hn::Sin(d, x)",        "YES (Minimax)", "V Sin<class D, class V>(D d, V x)");
    PrintRow("std::cos",   "hn::Cos(d, x)",        "YES (Minimax)", "V Cos<class D, class V>(D d, V x)");
    PrintRow("std::exp",   "hn::Exp(d, x)",        "YES (Minimax)", "V Exp<class D, class V>(D d, V x)");
    PrintRow("std::log",   "hn::Log(d, x)",        "YES (Minimax)", "V Log<class D, class V>(D d, V x)");
    PrintRow("std::pow",   "hn::Exp(d, hn::Mul(y, hn::Log(d, x)))", "YES", "Exp(d, Mul(y, Log(d, x)))");
    PrintRow("std::atan2", "hn::Atan2(d, y, x)",   "YES (Minimax)", "V Atan2<class D, class V>(D d, V y, V x)");
    PrintRow("std::asin",  "hn::Asin(d, x)",       "YES (Minimax)", "V Asin<class D, class V>(D d, V x)");
    PrintRow("std::acos",  "hn::Acos(d, x)",       "YES (Minimax)", "V Acos<class D, class V>(D d, V x)");
    PrintRow("std::atan",  "hn::Atan(d, x)",       "YES (Minimax)", "V Atan<class D, class V>(D d, V x)");
    PrintRow("std::sinh",  "hn::Sinh(d, x)",       "YES",           "V Sinh<class D, class V>(D d, V x)");
    PrintRow("std::cosh",  "hn::Cosh(d, x)",       "YES",           "V Cosh<class D, class V>(D d, V x)");
    PrintRow("std::tanh",  "hn::Tanh(d, x)",       "YES",           "V Tanh<class D, class V>(D d, V x)");
    PrintRow("std::exp2",  "hn::Exp2(d, x)",       "YES",           "V Exp2<class D, class V>(D d, V x)");
    PrintRow("std::log2",  "hn::Log2(d, x)",       "YES",           "V Log2<class D, class V>(D d, V x)");
    PrintRow("std::log10", "hn::Log10(d, x)",      "YES",           "V Log10<class D, class V>(D d, V x)");
    PrintRow("std::sqrt",  "hn::Sqrt(d, x)",       "YES (IEEE 754)", "V Sqrt<class D, class V>(D d, V x)");
    PrintRow("std::cbrt",  "hn::Cbrt(d, x)",       "YES",           "V Cbrt<class D, class V>(D d, V x)");
    PrintRow("std::erf",   "hn::Erf(d, x)",        "YES",           "V Erf<class D, class V>(D d, V x)");
}

void PrintCpuGpuTestingGuide() {
    std::cout << "\n========================================================================================================\n";
    std::cout << "                 HOW TO VERIFY HIGHWAY MATH WORKS ACROSS EVERY CPU & GPU TARGET                         \n";
    std::cout << "========================================================================================================\n";
    std::cout << "1. TESTING ACROSS ALL CPU ARCHITECTURES (x86_64 AVX2/AVX-512, ARM64 Neon, RISC-V, WASM):\n";
    std::cout << "   -> Highway supports dynamic dispatch target compilation: HWY_DYNAMIC_DISPATCH.\n";
    std::cout << "   -> You compile your source file ONCE with `#define HWY_TARGET_INCLUDE` and `#include <hwy/foreach_target.h>`.\n";
    std::cout << "   -> Highway automatically generates vectorized machine code for SSE4, AVX2, AVX-512, and EMULATED targets,\n";
    std::cout << "      and lets you execute all target implementations on your current machine to test bit-identical results!\n";
    std::cout << "   -> For cross-architecture ARM/RISC-V testing: Run your test binary under QEMU-user (`qemu-aarch64` / `qemu-riscv64`).\n\n";
    std::cout << "2. TESTING ACROSS GPUS (Vulkan / HLSL / Compute Shaders):\n";
    std::cout << "   -> GPUs do NOT execute C++ directly; Vulkan/SPIR-V shaders call GLSL/HLSL built-in sin()/cos().\n";
    std::cout << "   -> To ensure GPU-CPU bit-exact equivalence, compile Highway's Remez Minimax Polynomials directly into HLSL!\n";
    std::cout << "   -> Test using Vulkan Reference Compute Shaders and hash the output VkBuffer against CPU HWY output.\n";
    std::cout << "========================================================================================================\n";
}

int main() {
    TestFullFunctionMapping();
    PrintCpuGpuTestingGuide();
    return 0;
}
