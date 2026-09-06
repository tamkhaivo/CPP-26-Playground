#pragma once
#include "FixedMath.hpp"
#include <vector>
#include <cstdint>

namespace Type0 {
namespace Testing {

struct Vertex {
    Vector4Fixed position; // ND-Coordinates or Screen Coordinates depending on stage
    uint32_t color;        // RGBA packed
};

class ReferenceRasterizer {
public:
    ReferenceRasterizer(uint32_t width, uint32_t height);

    void Clear(uint32_t clearColor, int32_t clearDepth = 0x7FFFFFFF);
    
    // Renders a triangle given 3 screen-space vertices
    // Note: Z depth buffering uses fixed-point values mapped to an integer buffer
    void DrawTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2);

    const std::vector<uint32_t>& GetColorBuffer() const { return m_colorBuffer; }
    uint32_t GetWidth() const { return m_width; }
    uint32_t GetHeight() const { return m_height; }

private:
    // Edge function for half-space rasterization using fixed point math
    static int32_t EdgeFunction(const Vector2Fixed& a, const Vector2Fixed& b, const Vector2Fixed& c);
    
    uint32_t m_width;
    uint32_t m_height;
    std::vector<uint32_t> m_colorBuffer;
    std::vector<int32_t> m_depthBuffer; // 32-bit integer depth
};

} // namespace Testing
} // namespace Type0
