#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cmath>

namespace std {
namespace experimental {

// Dummy tags and ABI
struct element_aligned_tag {};
inline constexpr element_aligned_tag element_aligned{};

template<typename T>
struct native_simd_mask;

template<typename T>
struct native_simd;

// Specialization for native_simd_mask<float>
template<>
struct native_simd_mask<float> {
    __m128 m;
    native_simd_mask() : m(_mm_setzero_ps()) {}
    native_simd_mask(__m128 mask) : m(mask) {}

    template<typename F>
    native_simd_mask(F f) {
        alignas(16) int m_val[4];
        for (size_t i = 0; i < 4; ++i) {
            m_val[i] = f(i) ? -1 : 0;
        }
        m = _mm_castsi128_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(m_val)));
    }
};

// Specialization for native_simd<int>
template<>
struct native_simd<int> {
    __m128i v;
    native_simd() : v(_mm_setzero_si128()) {}
    native_simd(__m128i val) : v(val) {}
    void copy_from(const int* ptr, element_aligned_tag) {
        v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    }
    void copy_to(int* ptr, element_aligned_tag) const {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), v);
    }
};

namespace simd_abi {
    template<typename T>
    using native = native_simd<T>;
}

template<typename T, typename Abi = simd_abi::native<T>>
using simd = native_simd<T>;

// Specialization for native_simd<float>
template<>
struct native_simd<float> {
    __m128 v;

    native_simd() : v(_mm_setzero_ps()) {}
    native_simd(float val) : v(_mm_set1_ps(val)) {}
    native_simd(__m128 val) : v(val) {}

    static constexpr size_t size() { return 4; }

    void copy_from(const float* ptr, element_aligned_tag) {
        v = _mm_loadu_ps(ptr);
    }

    void copy_from(const float* ptr, native_simd_mask<float> mask, element_aligned_tag) {
        __m128 loaded = _mm_loadu_ps(ptr);
        v = _mm_blendv_ps(_mm_setzero_ps(), loaded, mask.m);
    }

    void copy_to(float* ptr, element_aligned_tag) const {
        _mm_storeu_ps(ptr, v);
    }

    void copy_to(float* ptr, native_simd_mask<float> mask, element_aligned_tag) const {
        alignas(16) float f_val[4];
        alignas(16) int m_val[4];
        _mm_store_ps(f_val, v);
        _mm_store_si128(reinterpret_cast<__m128i*>(m_val), _mm_castps_si128(mask.m));
        for (size_t i = 0; i < 4; ++i) {
            if (m_val[i]) {
                ptr[i] = f_val[i];
            }
        }
    }

    // Gather Constructor
    native_simd(const float* src, native_simd<int> idx) {
        alignas(16) int indices[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(indices), idx.v);
        v = _mm_setr_ps(src[indices[0]], src[indices[1]], src[indices[2]], src[indices[3]]);
    }

    // Scatter copy_to
    void copy_to(float* dest, native_simd<int> idx) const {
        alignas(16) float values[4];
        alignas(16) int indices[4];
        _mm_store_ps(values, v);
        _mm_store_si128(reinterpret_cast<__m128i*>(indices), idx.v);
        dest[indices[0]] = values[0];
        dest[indices[1]] = values[1];
        dest[indices[2]] = values[2];
        dest[indices[3]] = values[3];
    }
};

// Operator overloads
inline native_simd<float> operator+(native_simd<float> a, native_simd<float> b) {
    return _mm_add_ps(a.v, b.v);
}
inline native_simd<float>& operator+=(native_simd<float>& a, native_simd<float> b) {
    a.v = _mm_add_ps(a.v, b.v);
    return a;
}
inline native_simd<float> operator*(native_simd<float> a, native_simd<float> b) {
    return _mm_mul_ps(a.v, b.v);
}

// Comparisons
inline native_simd_mask<float> operator>(native_simd<float> a, native_simd<float> b) {
    return _mm_cmpgt_ps(a.v, b.v);
}

// Reduction
inline float reduce(native_simd<float> val) {
    alignas(16) float f[4];
    _mm_store_ps(f, val.v);
    return f[0] + f[1] + f[2] + f[3];
}

// Proxy for where clause
struct where_proxy {
    native_simd_mask<float> mask;
    native_simd<float>& vout;

    where_proxy(native_simd_mask<float> m, native_simd<float>& v) : mask(m), vout(v) {}

    void operator=(native_simd<float> vin) {
        vout.v = _mm_blendv_ps(vout.v, vin.v, mask.m);
    }
};

inline where_proxy where(native_simd_mask<float> mask, native_simd<float>& vout) {
    return where_proxy(mask, vout);
}

} // namespace experimental
} // namespace std
