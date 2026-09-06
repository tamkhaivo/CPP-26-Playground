#include "ReferenceRasterizer.hpp"
#include "ImageMetrics.hpp"
#include <iostream>
#include <vector>

using namespace Type0::Testing;

// Mock function representing the GPU backend (Vulkan/DX) rendering the same scene
// In reality, this would initialize the engine, run a frame, and read back the framebuffer from VRAM.
std::vector<uint32_t> MockGPURenderer(uint32_t width, uint32_t height) {
    std::vector<uint32_t> gpuBuffer(width * height, 0xFF000000);
    
    // Simulate slight floating point differences by drawing a slightly larger/shifted triangle
    // (We just manually write a few pixels for this mock example)
    for (uint32_t y = 30; y < 225; ++y) {
        for (uint32_t x = 30; x < 225; ++x) {
            if (x > y) {
                gpuBuffer[y * width + x] = 0xFF0000FF; // Red
            }
        }
    }
    return gpuBuffer;
}

void RunVisualSimilarityTest() {
    uint32_t width = 256;
    uint32_t height = 256;

    // 1. Generate CPU Reference Image
    ReferenceRasterizer rasterizer(width, height);
    rasterizer.Clear(0xFF000000);
    
    Vertex v0{ {Fixed16(128.0f), Fixed16(32.0f), Fixed16(0.5f), Fixed16(1.0f)}, 0xFF0000FF }; 
    Vertex v1{ {Fixed16(224.0f), Fixed16(224.0f), Fixed16(0.5f), Fixed16(1.0f)}, 0xFF0000FF }; 
    Vertex v2{ {Fixed16(32.0f), Fixed16(224.0f), Fixed16(0.5f), Fixed16(1.0f)}, 0xFF0000FF }; 
    rasterizer.DrawTriangle(v0, v1, v2);

    const auto& cpuImage = rasterizer.GetColorBuffer();

    // 2. Read back GPU rendered image
    std::vector<uint32_t> gpuImage = MockGPURenderer(width, height);

    // 3. Compare with relaxed tolerance
    double mse = ImageMetrics::CalculateMSE(cpuImage, gpuImage);
    
    // We allow some Mean Squared Error because GPU floating point math and rasterization rules vary
    constexpr double MSE_TOLERANCE = 50.0; 

    std::cout << "GPU Visual Similarity Test:\n";
    std::cout << "Calculated MSE: " << mse << "\n";
    std::cout << "Tolerance:      " << MSE_TOLERANCE << "\n";

    if (mse <= MSE_TOLERANCE) {
        std::cout << "STATUS: PASSED (Visually Similar)\n";
    } else {
        std::cout << "STATUS: FAILED (MSE exceeds tolerance)\n";
    }
}

int main() {
    RunVisualSimilarityTest();
    return 0;
}
