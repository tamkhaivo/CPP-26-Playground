#pragma once
#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>
#include <cstdint>
#include <cmath>

namespace Type0 {
namespace Math {

namespace hn = hwy::HWY_NAMESPACE;

// ============================================================================
// Hermetic Highway Transcendental Math Engine
// Guaranteed identical floating-point bitwise results across CPU architectures
// by strictly evaluating Highway's vector polynomial minimax implementations.
// ============================================================================

struct HermeticMath {
    // Hermetic SIMD Vector Sin: sin(x)
    template <class D, class V>
    HWY_INLINE static V Sin(D d, V x) {
        return hn::Sin(d, x);
    }

    // Hermetic SIMD Vector Cos: cos(x)
    template <class D, class V>
    HWY_INLINE static V Cos(D d, V x) {
        return hn::Cos(d, x);
    }

    // Hermetic SIMD Vector Exp: exp(x)
    template <class D, class V>
    HWY_INLINE static V Exp(D d, V x) {
        return hn::Exp(d, x);
    }

    // Hermetic SIMD Vector Log: log(x)
    template <class D, class V>
    HWY_INLINE static V Log(D d, V x) {
        return hn::Log(d, x);
    }

    // Hermetic SIMD Vector Atan2: atan2(y, x)
    template <class D, class V>
    HWY_INLINE static V Atan2(D d, V y, V x) {
        return hn::Atan2(d, y, x);
    }

    // Hermetic SIMD Vector Pow: pow(x, y) = exp(y * log(x))
    template <class D, class V>
    HWY_INLINE static V Pow(D d, V x, V y) {
        // pow(x, y) = exp(y * log(x))
        auto log_x = hn::Log(d, x);
        auto y_log_x = hn::Mul(y, log_x);
        return hn::Exp(d, y_log_x);
    }

    // Scalar fallback wrappers for individual float values using 1-lane Highway vectors
    HWY_INLINE static float Sin(float x) {
        const hn::ScalableTag<float> d;
        auto vx = hn::Set(d, x);
        auto vr = hn::Sin(d, vx);
        return hn::GetLane(vr);
    }

    HWY_INLINE static float Cos(float x) {
        const hn::ScalableTag<float> d;
        auto vx = hn::Set(d, x);
        auto vr = hn::Cos(d, vx);
        return hn::GetLane(vr);
    }

    HWY_INLINE static float Exp(float x) {
        const hn::ScalableTag<float> d;
        auto vx = hn::Set(d, x);
        auto vr = hn::Exp(d, vx);
        return hn::GetLane(vr);
    }

    HWY_INLINE static float Log(float x) {
        const hn::ScalableTag<float> d;
        auto vx = hn::Set(d, x);
        auto vr = hn::Log(d, vx);
        return hn::GetLane(vr);
    }

    HWY_INLINE static float Atan2(float y, float x) {
        const hn::ScalableTag<float> d;
        auto vy = hn::Set(d, y);
        auto vx = hn::Set(d, x);
        auto vr = hn::Atan2(d, vy, vx);
        return hn::GetLane(vr);
    }

    HWY_INLINE static float Pow(float x, float y) {
        const hn::ScalableTag<float> d;
        auto vx = hn::Set(d, x);
        auto vy = hn::Set(d, y);
        auto vr = Pow(d, vx, vy);
        return hn::GetLane(vr);
    }
};

} // namespace Math
} // namespace Type0
