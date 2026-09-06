#pragma once
#include <vector>
#include <cstdint>
#include <cmath>

namespace Type0 {
namespace Testing {

class ImageMetrics {
public:
    // Simple but deterministic FNV-1a hash over the color buffer
    static uint64_t CalculateHash(const std::vector<uint32_t>& colorBuffer) {
        uint64_t hash = 14695981039346656037ULL;
        for (uint32_t pixel : colorBuffer) {
            // Hash each byte
            for (int i = 0; i < 4; ++i) {
                uint8_t byte = (pixel >> (i * 8)) & 0xFF;
                hash ^= byte;
                hash *= 1099511628211ULL;
            }
        }
        return hash;
    }

    // Calculates the Mean Squared Error between two images of the same dimensions
    static double CalculateMSE(const std::vector<uint32_t>& imgA, const std::vector<uint32_t>& imgB) {
        if (imgA.size() != imgB.size() || imgA.empty()) return -1.0;

        double sumSquaredError = 0.0;
        for (size_t i = 0; i < imgA.size(); ++i) {
            uint32_t pA = imgA[i];
            uint32_t pB = imgB[i];

            for (int j = 0; j < 4; ++j) {
                int32_t cA = (pA >> (j * 8)) & 0xFF;
                int32_t cB = (pB >> (j * 8)) & 0xFF;
                int32_t diff = cA - cB;
                sumSquaredError += diff * diff;
            }
        }

        return sumSquaredError / (imgA.size() * 4.0); // 4 channels per pixel
    }
};

} // namespace Testing
} // namespace Type0
