#include "ReferenceRasterizer.hpp"
#include <algorithm>

namespace Type0 {
namespace Testing {

ReferenceRasterizer::ReferenceRasterizer(uint32_t width, uint32_t height)
    : m_width(width), m_height(height) {
    m_colorBuffer.resize(width * height, 0);
    m_depthBuffer.resize(width * height, 0x7FFFFFFF);
}

void ReferenceRasterizer::Clear(uint32_t clearColor, int32_t clearDepth) {
    std::fill(m_colorBuffer.begin(), m_colorBuffer.end(), clearColor);
    std::fill(m_depthBuffer.begin(), m_depthBuffer.end(), clearDepth);
}

// Edge function calculates a signed area (in fixed point representation).
// E(P) = (c.x - a.x)*(b.y - a.y) - (c.y - a.y)*(b.x - a.x)
int32_t ReferenceRasterizer::EdgeFunction(const Vector2Fixed& a, const Vector2Fixed& b, const Vector2Fixed& c) {
    // We compute this in raw integer to avoid dropping precision if we used fixed-point multiply
    // Since coordinates are 16.16, multiplying them gives 32.32, which we can keep in 64-bit int.
    int64_t ax = a.x.v; int64_t ay = a.y.v;
    int64_t bx = b.x.v; int64_t by = b.y.v;
    int64_t cx = c.x.v; int64_t cy = c.y.v;

    int64_t result = (cx - ax) * (by - ay) - (cy - ay) * (bx - ax);
    // Shift down to fit inside 32-bit (assuming reasonably sized screen coords)
    return static_cast<int32_t>(result >> 16); 
}

void ReferenceRasterizer::DrawTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2) {
    Vector2Fixed p0{v0.position.x, v0.position.y};
    Vector2Fixed p1{v1.position.x, v1.position.y};
    Vector2Fixed p2{v2.position.x, v2.position.y};

    // Bounding box (integer pixels)
    int32_t minX = std::max(0, std::min({p0.x.GetInt(), p1.x.GetInt(), p2.x.GetInt()}));
    int32_t minY = std::max(0, std::min({p0.y.GetInt(), p1.y.GetInt(), p2.y.GetInt()}));
    int32_t maxX = std::min(static_cast<int32_t>(m_width - 1), std::max({p0.x.GetInt() + 1, p1.x.GetInt() + 1, p2.x.GetInt() + 1}));
    int32_t maxY = std::min(static_cast<int32_t>(m_height - 1), std::max({p0.y.GetInt() + 1, p1.y.GetInt() + 1, p2.y.GetInt() + 1}));

    // Area of the triangle (for barycentric normalization)
    int32_t area = EdgeFunction(p0, p1, p2);
    if (area <= 0) return; // Backface culling or degenerate triangle

    for (int32_t y = minY; y <= maxY; ++y) {
        for (int32_t x = minX; x <= maxX; ++x) {
            // Pixel center in fixed point
            Vector2Fixed p{Fixed16(static_cast<float>(x) + 0.5f), Fixed16(static_cast<float>(y) + 0.5f)};

            int32_t w0 = EdgeFunction(p1, p2, p);
            int32_t w1 = EdgeFunction(p2, p0, p);
            int32_t w2 = EdgeFunction(p0, p1, p);

            // Inside triangle check (Top-Left fill rule not fully implemented for simplicity, just checking > 0)
            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                // Barycentric coordinates
                // We use 64-bit integer division to interpolate depth safely
                int64_t z0 = v0.position.z.v;
                int64_t z1 = v1.position.z.v;
                int64_t z2 = v2.position.z.v;

                int64_t z = (z0 * w0 + z1 * w1 + z2 * w2) / area;
                int32_t depth = static_cast<int32_t>(z);

                uint32_t idx = y * m_width + x;
                if (depth < m_depthBuffer[idx]) {
                    m_depthBuffer[idx] = depth;
                    
                    // Simple flat shading for now (taking color from first vertex)
                    m_colorBuffer[idx] = v0.color; 
                }
            }
        }
    }
}

} // namespace Testing
} // namespace Type0
