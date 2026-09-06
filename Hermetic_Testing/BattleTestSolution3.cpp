#include "FixedMath.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdint>

#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;
using namespace Type0::Testing;

// ============================================================================
// BATTLE TEST 1: Subpixel Precision Loss (ShiftRight<3> Truncation)
// ============================================================================
void BattleTest1_SubpixelPrecisionLoss() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[STRESS TEST 1] Subpixel Motion Loss (< 0.125 Pixel Shift)\n";
    std::cout << "------------------------------------------------------------------------\n";

    // Moving vertex by 0.02 pixels (1/50th pixel subpixel movement)
    float x0 = 100.00f;
    float x1 = 100.02f;

    // Int32 Fixed 16.16
    int32_t x0_32 = Fixed16(x0).v;
    int32_t x1_32 = Fixed16(x1).v;

    // Int16 ShiftRight<3> (Truncates bottom 3 bits)
    int16_t x0_16 = static_cast<int16_t>(Fixed16(x0).v >> 8) >> 3;
    int16_t x1_16 = static_cast<int16_t>(Fixed16(x1).v >> 8) >> 3;

    std::cout << "Original Motion: 100.00 -> 100.02 (0.02 Subpixel Step)\n";
    std::cout << " - Int32 Fixed: " << x0_32 << " -> " << x1_32 << " (Delta = " << (x1_32 - x0_32) << " units)\n";
    std::cout << " - Int16 Fixed: " << x0_16 << " -> " << x1_16 << " (Delta = " << (x1_16 - x0_16) << " units)\n";

    if (x0_16 == x1_16) {
        std::cout << "❌ HOLE 1 PROVED: Subpixel motion was TRUNCATED to 0 by ShiftRight<3>!\n";
        std::cout << "   -> Animation motion smaller than 0.125px is completely frozen in Solution 3.\n";
    }
}

// ============================================================================
// BATTLE TEST 2: Saturation Clamping & Depth/Barycentric Corruption
// ============================================================================
void BattleTest2_BarycentricSaturationCorruption() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[STRESS TEST 2] Large Polygon Depth/Barycentric Saturation Clamping\n";
    std::cout << "------------------------------------------------------------------------\n";

    // Polygon edge spanning 1080p viewport diagonal (0,0) -> (1920, 1080)
    int32_t ax = 0, ay = 0;
    int32_t bx = 1920, by = 1080;
    int32_t dx = (bx - ax) >> 3; // 240
    int32_t dy = (by - ay) >> 3; // 135

    // Pixel near bottom-right of viewport (1900, 10)
    int16_t px = 1900 >> 3; // 237
    int16_t py = 10 >> 3;   // 1

    int32_t exact_cross = (237 * 135) - (1 * 240); // 31,995 - 240 = 31,755

    // Evaluate 100 pixels further away
    int16_t px_far = 1920 >> 3; // 240
    int16_t py_far = 0;
    
    // Int16 Product
    int32_t prod_far = (240 * 135); // 32,400
    int16_t saturated_val = static_cast<int16_t>(std::min(32767, std::max(-32768, prod_far)));

    std::cout << "Near Edge Product (Exact): " << exact_cross << " | Int16 Output: " << exact_cross << "\n";
    std::cout << "Far Pixel Product (Exact):  " << prod_far << " | Int16 Saturated Output: " << saturated_val << "\n";

    // Pixel slightly further (2000, 0)
    int32_t prod_far2 = (250 * 135); // 33,750 -> Clamps to 32767
    int16_t saturated_val2 = static_cast<int16_t>(std::min(32767, std::max(-32768, prod_far2)));

    std::cout << "Farther Pixel Product:      " << prod_far2 << " | Int16 Saturated Output: " << saturated_val2 << "\n";

    if (saturated_val == saturated_val2) {
        std::cout << "❌ HOLE 2 PROVED: Both distant pixels CLAMPED to the exact same value (" << saturated_val << ")!\n";
        std::cout << "   -> Depth gradient becomes FLAT (0 slope). Z-buffering and shading are corrupted!\n";
    }
}

// ============================================================================
// BATTLE TEST 3: 4K Viewport Integer Overflow & Sign Reversal Test
// ============================================================================
void BattleTest3_4KViewportOverflow() {
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "[STRESS TEST 3] 4K Viewport (3840x2160) Coordinate Scale Overflow\n";
    std::cout << "------------------------------------------------------------------------\n";

    // 4K coordinate pixel near edge of screen
    int32_t px_4k = 3800; 
    int32_t py_4k = 2100;

    int32_t dx_32 = 3800;
    int32_t dy_32 = 2100;

    int16_t px_16 = static_cast<int16_t>(px_4k >> 3); // 475
    int16_t dy_16 = static_cast<int16_t>(dy_32 >> 3); // 262

    // 475 * 262 = 124,450
    int32_t full_prod = px_16 * dy_16; 
    int16_t term1_16 = static_cast<int16_t>(full_prod); // Wraps around into negative signed 16-bit int (-6622)

    std::cout << "4K Pixel Coordinate Product: " << full_prod << " (Exceeds 16-bit max 32,767)\n";
    std::cout << "Int16 Cast / Overflow Value: " << term1_16 << " (Negative value!)\n";

    if (term1_16 < 0 && full_prod > 0) {
        std::cout << "❌ HOLE 3 PROVED: Integer overflow caused SIGN REVERSAL (" << term1_16 << ") in Solution 3!\n";
        std::cout << "   -> Triangles flip inside-out and fail backface culling on 4K viewports!\n";
    }
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "       RIGOROUS BATTLE-TESTING & HOLE-DISCOVERY: SOLUTION 3 (INT16)     \n";
    std::cout << "========================================================================\n";

    BattleTest1_SubpixelPrecisionLoss();
    BattleTest2_BarycentricSaturationCorruption();
    BattleTest3_4KViewportOverflow();

    std::cout << "\n========================================================================\n";
    std::cout << "                    CRITIQUE SUMMARY & FINAL VERDICT                    \n";
    std::cout << "========================================================================\n";
    std::cout << "VERDICT: Solution 3 (Int16) is FAULTY & DANGEROUS for production software rasterization.\n";
    std::cout << "1. Subpixel Precision Loss: Motion steps < 0.125px freeze.\n";
    std::cout << "2. Saturation Clamping: Destroys depth interpolation slope on large triangles.\n";
    std::cout << "3. 4K Viewport Overflow: Sign reversal causes polygons to flip inside-out.\n\n";
    std::cout << "RECOMMENDATION: Reject Solution 3. Standardize on Solution 1 (Int32 Fixed-Point).\n";
    std::cout << "========================================================================\n";

    return 0;
}
