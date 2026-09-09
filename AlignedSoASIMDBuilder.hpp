#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <hwy/highway.h>

namespace Type0::ECS {

// ============================================================================
// 1. Cross-Platform 64-Byte Cache Line Aligned Allocator
// ============================================================================
template <typename T, size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}


    [[nodiscard]] T* allocate(size_t n) {
        size_t bytes = n * sizeof(T);
        void* ptr = nullptr;
#if defined(_WIN32)
        ptr = _aligned_malloc(bytes, Alignment);
#else
        if (posix_memalign(&ptr, Alignment, bytes) != 0) ptr = nullptr;
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t) noexcept {
        if (!p) return;
#if defined(_WIN32)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const AlignedAllocator<U>&) const noexcept { return false; }
};


template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

// ============================================================================
// 2. Aligned SoA Component Storage
// ============================================================================
struct alignas(64) PositionSoA {
    AlignedVector<float> x;
    AlignedVector<float> y;
    AlignedVector<float> z;

    void Reserve(size_t capacity) {
        x.reserve(capacity);
        y.reserve(capacity);
        z.reserve(capacity);
    }

    [[nodiscard]] size_t Size() const noexcept { return x.size(); }

    // Enforce SIMD tail padding to prevent vector register out-of-bounds reads/writes
    void EnforceTailPadding() {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<float> d;
        const size_t lanes = hn::Lanes(d);
        size_t remainder = x.size() % lanes;

        if (remainder != 0) {
            size_t padCount = lanes - remainder;
            for (size_t i = 0; i < padCount; ++i) {
                x.push_back(0.0f);
                y.push_back(0.0f);
                z.push_back(0.0f);
            }
        }
    }
};

struct alignas(64) VelocitySoA {
    AlignedVector<float> vx;
    AlignedVector<float> vy;
    AlignedVector<float> vz;

    void Reserve(size_t capacity) {
        vx.reserve(capacity);
        vy.reserve(capacity);
        vz.reserve(capacity);
    }

    [[nodiscard]] size_t Size() const noexcept { return vx.size(); }

    void EnforceTailPadding() {
        namespace hn = hwy::HWY_NAMESPACE;
        const hn::ScalableTag<float> d;
        const size_t lanes = hn::Lanes(d);
        size_t remainder = vx.size() % lanes;

        if (remainder != 0) {
            size_t padCount = lanes - remainder;
            for (size_t i = 0; i < padCount; ++i) {
                vx.push_back(0.0f);
                vy.push_back(0.0f);
                vz.push_back(0.0f);
            }
        }
    }
};

// ============================================================================
// 3. Aligned SoA Entity Builder
// ============================================================================
class AlignedSoABuilder {
private:
    PositionSoA& m_positions;
    VelocitySoA& m_velocities;
    size_t m_stagedIndex;

public:
    AlignedSoABuilder(PositionSoA& pos, VelocitySoA& vel)
        : m_positions(pos), m_velocities(vel), m_stagedIndex(pos.Size()) {}

    ~AlignedSoABuilder() {
        Finalize();
    }

    void Finalize() {
        m_positions.EnforceTailPadding();
        m_velocities.EnforceTailPadding();
    }

    AlignedSoABuilder& WithPosition(float x, float y, float z) {
        m_positions.x.push_back(x);
        m_positions.y.push_back(y);
        m_positions.z.push_back(z);
        return *this;
    }

    AlignedSoABuilder& WithVelocity(float vx, float vy, float vz) {
        m_velocities.vx.push_back(vx);
        m_velocities.vy.push_back(vy);
        m_velocities.vz.push_back(vz);
        return *this;
    }

    [[nodiscard]] uint32_t Build() {
        return static_cast<uint32_t>(m_stagedIndex++);
    }
};

// ============================================================================
// 4. Batch Archetype Builder
// ============================================================================
class ArchetypeBatchBuilder {
private:
    PositionSoA& m_positions;
    VelocitySoA& m_velocities;

public:
    ArchetypeBatchBuilder(PositionSoA& pos, VelocitySoA& vel)
        : m_positions(pos), m_velocities(vel) {}

    void SpawnBatch(size_t count, float posX, float posY, float posZ, float velX, float velY, float velZ) {
        m_positions.Reserve(m_positions.Size() + count);
        m_velocities.Reserve(m_velocities.Size() + count);

        for (size_t i = 0; i < count; ++i) {
            m_positions.x.push_back(posX);
            m_positions.y.push_back(posY);
            m_positions.z.push_back(posZ);

            m_velocities.vx.push_back(velX);
            m_velocities.vy.push_back(velY);
            m_velocities.vz.push_back(velZ);
        }

        m_positions.EnforceTailPadding();
        m_velocities.EnforceTailPadding();
    }
};

// ============================================================================
// 5. Zero-Copy SIMD Arena Builder (Host-to-GPU Push Buffer)
// ============================================================================
class SIMDArenaBuilder {
private:
    uint8_t* m_basePtr{nullptr};
    size_t m_offset{0};
    size_t m_capacity{0};

public:
    SIMDArenaBuilder(void* memoryAddress, size_t capacityBytes)
        : m_basePtr(reinterpret_cast<uint8_t*>(memoryAddress)), m_capacity(capacityBytes) {
        std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(m_basePtr);
        size_t pad = (64 - (addr & 63)) & 63;
        if (pad > 0) {
            if (m_capacity < pad) {
                throw std::runtime_error("Insufficient buffer capacity for alignment padding");
            }
            m_basePtr += pad;
            m_capacity -= pad;
        }
    }

    template <typename T>
    SIMDArenaBuilder& PushSIMDBlock(const T* sourceData, size_t count) {
        size_t bytes = count * sizeof(T);
        size_t alignedBytes = (bytes + 63) & ~size_t(63); // Round up to 64 bytes

        if (m_offset + alignedBytes > m_capacity) {
            throw std::runtime_error("SIMDArenaBuilder buffer overflow");
        }

        std::memcpy(m_basePtr + m_offset, sourceData, bytes);
        if (alignedBytes > bytes) {
            std::memset(m_basePtr + m_offset + bytes, 0, alignedBytes - bytes);
        }
        m_offset += alignedBytes;
        return *this;
    }

    [[nodiscard]] const void* GetBufferHandle() const noexcept { return m_basePtr; }
    [[nodiscard]] size_t GetBytesWritten() const noexcept { return m_offset; }
};

// ============================================================================
// 6. Vectorized SIMD Pipeline Execution (Highway SIMD Integration)
// ============================================================================
inline void SystemUpdateSIMDPhysics(PositionSoA& pos, const VelocitySoA& vel, float dt) {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<float> d;
    const auto vDt = hn::Set(d, dt);
    const size_t lanes = hn::Lanes(d);

    const size_t elementCount = std::min(pos.x.size(), vel.vx.size());
    size_t i = 0;
    for (; i + lanes <= elementCount; i += lanes) {
        const auto px = hn::Load(d, &pos.x[i]);
        const auto py = hn::Load(d, &pos.y[i]);
        const auto pz = hn::Load(d, &pos.z[i]);

        const auto vx = hn::Load(d, &vel.vx[i]);
        const auto vy = hn::Load(d, &vel.vy[i]);
        const auto vz = hn::Load(d, &vel.vz[i]);

        const auto newX = hn::MulAdd(vx, vDt, px);
        const auto newY = hn::MulAdd(vy, vDt, py);
        const auto newZ = hn::MulAdd(vz, vDt, pz);

        hn::Store(newX, d, &pos.x[i]);
        hn::Store(newY, d, &pos.y[i]);
        hn::Store(newZ, d, &pos.z[i]);
    }

    for (; i < elementCount; ++i) {
        pos.x[i] += vel.vx[i] * dt;
        pos.y[i] += vel.vy[i] * dt;
        pos.z[i] += vel.vz[i] * dt;
    }
}

} // namespace Type0::ECS
