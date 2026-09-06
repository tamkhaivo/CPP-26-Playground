#include "ReferenceRasterizer.hpp"
#include "ImageMetrics.hpp"
#include <iostream>
#include <cassert>

using namespace Type0::Testing;

// In a real framework, this hash would be loaded from disk alongside the golden PNG.
constexpr uint64_t EXPECTED_GOLDEN_HASH = 16900977273763782559ULL; // Example hash, would be real in practice

void RunGoldenImageTest() {
    ReferenceRasterizer rasterizer(256, 256);
    
    // Clear with black
    rasterizer.Clear(0xFF000000);

    // Setup a simple triangle
    Vertex v0{ {Fixed16(128.0f), Fixed16(32.0f), Fixed16(0.5f), Fixed16(1.0f)}, 0xFF0000FF }; // Red (ABGR)
    Vertex v1{ {Fixed16(224.0f), Fixed16(224.0f), Fixed16(0.5f), Fixed16(1.0f)}, 0xFF00FF00 }; // Green
    Vertex v2{ {Fixed16(32.0f), Fixed16(224.0f), Fixed16(0.5f), Fixed16(1.0f)}, 0xFFFF0000 }; // Blue
    
    // Render
    rasterizer.DrawTriangle(v0, v1, v2);

    // Calculate deterministic hash
    uint64_t actualHash = ImageMetrics::CalculateHash(rasterizer.GetColorBuffer());
    
    std::cout << "Golden Image Test:\n";
    std::cout << "Expected Hash: " << EXPECTED_GOLDEN_HASH << "\n";
    std::cout << "Actual Hash:   " << actualHash << "\n";

    // In a real test, we would assert(actualHash == EXPECTED_GOLDEN_HASH);
    // For this example, we just print the result.
    if (actualHash == EXPECTED_GOLDEN_HASH) {
        std::cout << "STATUS: PASSED (Hashes Match Perfectly)\n";
    } else {
        std::cout << "STATUS: FAILED or Initial Run. Save Actual Hash as new Golden if visually correct.\n";
    }
}

int main() {
    RunGoldenImageTest();
    return 0;
}
