#pragma once

#include <cmath>
#include <cstdint>
#include <concepts>
#include <algorithm>
#include <vector>
#include <array>
#include <cstring>
#include <hwy/highway.h>

namespace Type0::Math {

// Layout guaranteed matching Vulkan std430 and push constant alignment
struct alignas(16) Vec4 {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float w{0.0f};

    constexpr Vec4() = default;
    constexpr Vec4(float x_, float y_, float z_, float w_ = 1.0f) : x(x_), y(y_), z(z_), w(w_) {}

    [[nodiscard]] const float* Data() const noexcept { return &x; }
    [[nodiscard]] float* Data() noexcept { return &x; }

    constexpr bool operator==(const Vec4& o) const noexcept {
        return x == o.x && y == o.y && z == o.z && w == o.w;
    }
};

static_assert(sizeof(Vec4) == 16, "Vec4 must be exactly 16 bytes for Vulkan std430 layout");
static_assert(alignof(Vec4) == 16, "Vec4 must be 16-byte aligned");

struct alignas(16) Mat4 {
    Vec4 cols[4];

    constexpr Mat4() {
        cols[0] = {1.0f, 0.0f, 0.0f, 0.0f};
        cols[1] = {0.0f, 1.0f, 0.0f, 0.0f};
        cols[2] = {0.0f, 0.0f, 1.0f, 0.0f};
        cols[3] = {0.0f, 0.0f, 0.0f, 1.0f};
    }

    [[nodiscard]] const float* Data() const noexcept { return cols[0].Data(); }
    [[nodiscard]] float* Data() noexcept { return cols[0].Data(); }


    // Column-major Matrix Vector Multiplication
    constexpr Vec4 operator*(const Vec4& v) const noexcept {
        return {
            cols[0].x * v.x + cols[1].x * v.y + cols[2].x * v.z + cols[3].x * v.w,
            cols[0].y * v.x + cols[1].y * v.y + cols[2].y * v.z + cols[3].y * v.w,
            cols[0].z * v.x + cols[1].z * v.y + cols[2].z * v.z + cols[3].z * v.w,
            cols[0].w * v.x + cols[1].w * v.y + cols[2].w * v.z + cols[3].w * v.w
        };
    }

    // Column-major Matrix Matrix Multiplication
    constexpr Mat4 operator*(const Mat4& b) const noexcept {
        Mat4 res;
        res.cols[0] = (*this) * b.cols[0];
        res.cols[1] = (*this) * b.cols[1];
        res.cols[2] = (*this) * b.cols[2];
        res.cols[3] = (*this) * b.cols[3];
        return res;
    }

    // Vulkan Clip Space Perspective (z: 0 to 1, inverted Y for Vulkan viewport)
    static Mat4 PerspectiveVK(float fovRadians, float aspect, float zNear, float zFar) {
        float tanHalfFov = std::tan(fovRadians * 0.5f);
        Mat4 m;
        std::memset(&m, 0, sizeof(Mat4));
        m.cols[0].x = 1.0f / (aspect * tanHalfFov);
        m.cols[1].y = -1.0f / tanHalfFov; // Vulkan inverted Y
        m.cols[2].z = zFar / (zNear - zFar);
        m.cols[2].w = -1.0f;
        m.cols[3].z = (zNear * zFar) / (zNear - zFar);
        return m;
    }
};

static_assert(sizeof(Mat4) == 64, "Mat4 must be exactly 64 bytes for Vulkan std430 layout");

struct alignas(16) Quat {
    float x{0.0f}, y{0.0f}, z{0.0f}, w{1.0f};

    constexpr Quat() = default;
    constexpr Quat(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}

    constexpr Quat operator*(const Quat& q) const noexcept {
        return {
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w,
            w * q.w - x * q.x - y * q.y - z * q.z
        };
    }
};

static_assert(sizeof(Quat) == 16, "Quat must be 16 bytes");

struct alignas(16) AABB {
    Vec4 minPos;
    Vec4 maxPos;
};

static_assert(sizeof(AABB) == 32, "AABB must be exactly 32 bytes for Vulkan std430 layout");

// Highway SIMD Accelerated Batch Operations
namespace SIMD {

inline void BatchMulAdd(const Vec4* HWY_RESTRICT inputA,
                        const Vec4* HWY_RESTRICT inputB,
                        float scale,
                        Vec4* HWY_RESTRICT output,
                        size_t count) {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const auto vScale = hn::Set(d, scale);

    const float* pA = reinterpret_cast<const float*>(inputA);
    const float* pB = reinterpret_cast<const float*>(inputB);
    float* pOut = reinterpret_cast<float*>(output);
    const size_t totalFloats = count * 4;

    size_t i = 0;
    for (; i + lanes <= totalFloats; i += lanes) {
        auto a = hn::LoadU(d, pA + i);
        auto b = hn::LoadU(d, pB + i);
        auto res = hn::MulAdd(a, vScale, b);
        hn::StoreU(res, d, pOut + i);
    }

    for (; i < totalFloats; ++i) {
        pOut[i] = pA[i] * scale + pB[i];
    }
}

inline void BatchMatVecTransform(const Mat4& mat,
                                 const Vec4* HWY_RESTRICT inVecs,
                                 Vec4* HWY_RESTRICT outVecs,
                                 size_t count) {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<float> d;

    // Transform batch of 4D vectors by 4x4 matrix
    for (size_t i = 0; i < count; ++i) {
        outVecs[i] = mat * inVecs[i];
    }
}

} // namespace SIMD

// Deterministic Hashing Utility for Verification
inline uint64_t ComputeDeterminismChecksum(const Vec4* data, size_t count) {
    uint64_t hash = 14695981039346656037ULL; // FNV-1a basis
    const uint8_t* bytePtr = reinterpret_cast<const uint8_t*>(data);
    const size_t totalBytes = count * sizeof(Vec4);

    for (size_t i = 0; i < totalBytes; ++i) {
        hash ^= bytePtr[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace Type0::Math
