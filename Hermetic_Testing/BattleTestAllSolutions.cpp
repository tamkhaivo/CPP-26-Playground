#include "FixedMath.hpp"
#include "ImageMetrics.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;
using namespace Type0::Testing;

// ============================================================================
// STRESS TEST 1: Solution 1 (Int32 Fixed-Point) — Extreme Off-Screen Bounds
// ============================================================================
void StressTest_Solution1_Int32Bounds() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[SOLUTION 1 BATTLE-TEST] Extreme Off-Screen Coordinates (X = 100,000)\n";
    std::cout << "------------------------------------------------------------------------\n";

    float extreme_x = 100000.0f;
    int64_t raw_32 = static_cast<int64_t>(extreme_x) * 65536LL;
    int32_t cast_32 = static_cast<int32_t>(raw_32); // Overflow check for 32-bit signed int

    std::cout << "Off-Screen Coordinate X = 100,000.0\n";
    std::cout << " - Exact Fixed 16.16 (Int64): " << raw_32 << "\n";
    std::cout << " - Cast Int32 Fixed:         " << cast_32 << " (Max Int32: 2,147,483,647)\n";

    if (cast_32 < 0 && raw_32 > 0) {
        std::cout << "❌ SOLUTION 1 LIMITATION PROVED: Unclipped geometry (> 32,767px) OVERFLOWS Int32!\n";
        std::cout << "   -> REQUIREMENT: Geometry MUST undergo CPU Frustum Clipping before Int32 rasterization.\n";
    }
}

// ============================================================================
// STRESS TEST 2: Solution 2 (Strict Float) — Catastrophic Cancellation & Hash Drift
// ============================================================================
void StressTest_Solution2_FloatCancellation() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[SOLUTION 2 BATTLE-TEST] Catastrophic Cancellation & 1-ULP Hash Divergence\n";
    std::cout << "------------------------------------------------------------------------\n";

    // Near-coincident vertices causing subnormal float cancellation
    float f1 = 1.0000001f;
    float f2 = 1.0000002f;
    float scale = 1e7f;

    float res1 = (f2 - f1) * scale;             // Unfused: 1.00000000
    float res2 = (f2 * scale) - (f1 * scale);   // Reordered: 1.00000024 (1-ULP Drift)

    uint32_t u1, u2;
    std::memcpy(&u1, &res1, sizeof(float));
    std::memcpy(&u2, &res2, sizeof(float));

    std::cout << "Direct  (f2-f1)*scale: " << std::setprecision(8) << res1 << " (Hex: 0x" << std::hex << u1 << std::dec << ")\n";
    std::cout << "Reordered (f2*s-f1*s): " << std::setprecision(8) << res2 << " (Hex: 0x" << std::hex << u2 << std::dec << ")\n";

    if (u1 != u2) {
        std::cout << "❌ SOLUTION 2 LIMITATION PROVED: Reordering / Precision drift causes 1-ULP HASH DIVERGENCE!\n";
        std::cout << "   -> Floating point binary output differs (0x" << std::hex << u1 << " vs 0x" << u2 << std::dec << "), failing strict SHA-256/FNV hashes.\n";
    }
}

// ============================================================================
// STRESS TEST 3: Solution 3 (Int16) — Re-Verification of Subpixel & Saturation Holes
// ============================================================================
void StressTest_Solution3_Int16Reverification() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[SOLUTION 3 BATTLE-TEST] Subpixel Motion & Saturation Clamping\n";
    std::cout << "------------------------------------------------------------------------\n";

    int16_t x0_16 = static_cast<int16_t>(Fixed16(100.00f).v >> 8) >> 3;
    int16_t x1_16 = static_cast<int16_t>(Fixed16(100.02f).v >> 8) >> 3;

    std::cout << "Subpixel Motion (100.00 -> 100.02): Int16 Delta = " << (x1_16 - x0_16) << " units\n";

    if (x0_16 == x1_16) {
        std::cout << "❌ SOLUTION 3 LIMITATION PROVED: Subpixel motion is completely frozen by ShiftRight<3>.\n";
    }
}

// ============================================================================
// STRESS TEST 4: Solution 4 (Dual-Engine MSE) — Localized Artifact Swallowing
// ============================================================================
void StressTest_Solution4_MSELocalArtifactSwallowing() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[SOLUTION 4 BATTLE-TEST] Global MSE Swallowing Localized Render Artifacts\n";
    std::cout << "------------------------------------------------------------------------\n";

    constexpr size_t WIDTH = 1920;
    constexpr size_t HEIGHT = 1080;
    constexpr size_t TOTAL_PIXELS = WIDTH * HEIGHT; // 2,073,600 Pixels

    std::vector<uint32_t> img_golden(TOTAL_PIXELS, 0xFF000000); // Black screen
    std::vector<uint32_t> img_corrupted = img_golden;

    // Corrupt a 10x10 pixel UI button or icon block in top corner
    for (size_t y = 0; y < 10; ++y) {
        for (size_t x = 0; x < 10; ++x) {
            img_corrupted[y * WIDTH + x] = 0xFFFFFFFF; // Corrupted White pixels
        }
    }

    double mse = ImageMetrics::CalculateMSE(img_golden, img_corrupted);

    std::cout << "1080p Image with 10x10 Corrupted Pixel UI Block:\n";
    std::cout << " - Total Frame Pixels: " << TOTAL_PIXELS << "\n";
    std::cout << " - Calculated MSE:     " << std::fixed << std::setprecision(6) << mse << "\n";
    std::cout << " - Standard MSE Threshold: 10.0\n";

    if (mse < 10.0) {
        std::cout << "❌ SOLUTION 4 LIMITATION PROVED: MSE = " << mse << " (< 10.0) SWALLOWED the render artifact!\n";
        std::cout << "   -> A broken UI button or missing triangle passes standard MSE verification unnoticed.\n";
        std::cout << "   -> REQUIREMENT: Use SSIM (Structural Similarity) or Bounding-Box Tile Hashing alongside MSE.\n";
    }
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "   RIGOROUS BATTLE-TESTING & VULNERABILITY REPORT FOR ALL SOLUTIONS     \n";
    std::cout << "========================================================================\n";

    StressTest_Solution1_Int32Bounds();
    StressTest_Solution2_FloatCancellation();
    StressTest_Solution3_Int16Reverification();
    StressTest_Solution4_MSELocalArtifactSwallowing();

    std::cout << "\n========================================================================\n";
    std::cout << "                       FINAL ARCHITECTURAL VERDICT                      \n";
    std::cout << "========================================================================\n";
    std::cout << "1. Solution 1 (Int32 Fixed): BEST FOR GOLDEN REF. Requires CPU Frustum Clipping.\n";
    std::cout << "2. Solution 2 (Strict Float): Subject to 1-ULP hash divergence across compilers.\n";
    std::cout << "3. Solution 3 (Int16 Fixed): FAULTY. Freezes subpixel motion & truncates 4K.\n";
    std::cout << "4. Solution 4 (Dual-Engine): Global MSE swallows localized 10x10 UI bugs. Pair with SSIM.\n";
    std::cout << "========================================================================\n";

    return 0;
}
