#pragma once
#include <cstdint>

namespace Type0 {
namespace Testing {

// A deterministic 16.16 fixed-point math type to avoid floating-point non-determinism
// across different CPU architectures and compilers during reference rendering.
class Fixed16 {
public:
    int32_t v;

    constexpr Fixed16() : v(0) {}
    constexpr explicit Fixed16(int32_t val, bool raw) : v(raw ? val : val << 16) {}
    constexpr explicit Fixed16(float f) : v(static_cast<int32_t>(f * 65536.0f)) {}

    // Factory for raw initialization
    static constexpr Fixed16 FromRaw(int32_t raw) { return Fixed16(raw, true); }

    constexpr int32_t GetInt() const { return v >> 16; }
    constexpr float GetFloat() const { return static_cast<float>(v) / 65536.0f; }

    constexpr Fixed16 operator+(const Fixed16& rhs) const { return FromRaw(v + rhs.v); }
    constexpr Fixed16 operator-(const Fixed16& rhs) const { return FromRaw(v - rhs.v); }
    
    constexpr Fixed16 operator*(const Fixed16& rhs) const {
        return FromRaw(static_cast<int32_t>((static_cast<int64_t>(v) * rhs.v) >> 16));
    }
    
    constexpr Fixed16 operator/(const Fixed16& rhs) const {
        return FromRaw(static_cast<int32_t>((static_cast<int64_t>(v) << 16) / rhs.v));
    }

    constexpr bool operator==(const Fixed16& rhs) const { return v == rhs.v; }
    constexpr bool operator!=(const Fixed16& rhs) const { return v != rhs.v; }
    constexpr bool operator<(const Fixed16& rhs) const { return v < rhs.v; }
    constexpr bool operator<=(const Fixed16& rhs) const { return v <= rhs.v; }
    constexpr bool operator>(const Fixed16& rhs) const { return v > rhs.v; }
    constexpr bool operator>=(const Fixed16& rhs) const { return v >= rhs.v; }
};

struct Vector2Fixed {
    Fixed16 x, y;
};

struct Vector3Fixed {
    Fixed16 x, y, z;
};

struct Vector4Fixed {
    Fixed16 x, y, z, w;
};

} // namespace Testing
} // namespace Type0
