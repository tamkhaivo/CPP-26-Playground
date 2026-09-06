#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cmath>

// --- Google Highway SIMD Integration ---
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

// Compression Tiers
enum class MeshCompressionTier : uint8_t {
    Raw_Float32 = 0,
    Packed_Norm16_HWY = 1, // Google Highway Vectorized & Hermetic
    Quantized_Compact_HWY = 2
};

#pragma pack(push, 1)
struct Type0AssetHeader {
    char magic[4] = {'T', '0', 'A', '1'};
    uint32_t version = 1;
    uint32_t assetType = 1;
    uint8_t compressionTier = 0;
    uint8_t reserved[3] = {0};
    uint32_t payloadSize = 0;
    uint8_t payloadHash[32] = {0};
};

struct Type0MeshHeader {
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint32_t vertexStride = 0;
    float boundingMin[3] = {0.0f, 0.0f, 0.0f};
    float boundingMax[3] = {0.0f, 0.0f, 0.0f};
};

struct QuantizedVertexHighQuality {
    uint16_t posX, posY, posZ;
    uint16_t unusedPadding;
    int16_t normOctX, normOctY;
    uint16_t u, v;
};

struct RawVertexFloat32 {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
};
#pragma pack(pop)

// Deterministic FNV-1a Hash
uint64_t HashBufferFNV1a(const uint8_t* data, size_t size) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// --- Google Highway Hermetic SIMD Quantization Engine ---
void QuantizePositionsHighway(const float* inPositions, uint16_t* outPositions, size_t count) {
    const hn::ScalableTag<float> dFloat;
    const hn::Rebind<int32_t, decltype(dFloat)> dInt;

    const size_t lanes = hn::Lanes(dFloat);
    const auto vMinVal = hn::Set(dFloat, -100.0f);
    const auto vRangeScale = hn::Set(dFloat, 1.0f / 200.0f * 65535.0f);
    const auto vZero = hn::Set(dFloat, 0.0f);
    const auto vMaxHWY = hn::Set(dFloat, 65535.0f);
    const auto vHalf = hn::Set(dFloat, 0.5f);

    size_t i = 0;
    for (; i + lanes <= count; i += lanes) {
        // Load SIMD lane positions
        auto vPos = hn::Load(dFloat, inPositions + i);
        // Hermetic normalized scaling: (pos - min) * scale + 0.5f
        auto vNorm = hn::MulAdd(hn::Sub(vPos, vMinVal), vRangeScale, vHalf);
        // Clamp deterministically to [0, 65535]
        auto vClamped = hn::Clamp(vNorm, vZero, vMaxHWY);
        // Convert SIMD float lanes to int32 lanes
        auto vInt = hn::ConvertTo(dInt, vClamped);

        alignas(16) int32_t temp[16];
        hn::Store(vInt, dInt, temp);
        for (size_t l = 0; l < lanes; ++l) {
            outPositions[i + l] = static_cast<uint16_t>(temp[l]);
        }
    }

    // Scalar fallback tail loop
    for (; i < count; ++i) {
        float norm = (inPositions[i] + 100.0f) / 200.0f * 65535.0f + 0.5f;
        norm = std::clamp(norm, 0.0f, 65535.0f);
        outPositions[i] = static_cast<uint16_t>(norm);
    }
}

void RunHighwaySIMDCookerBenchmark() {
    std::cout << "\n=========================================================\n";
    std::cout << "  GOOGLE HIGHWAY SIMD HERMETIC ASSET COOKER BENCHMARK\n";
    std::cout << "=========================================================\n";

    const size_t testVertexCount = 100000;
    std::vector<RawVertexFloat32> rawVertices(testVertexCount);
    std::vector<float> posBuffer(testVertexCount * 3);

    for (size_t i = 0; i < testVertexCount; ++i) {
        float t = static_cast<float>(i);
        rawVertices[i] = {
            std::sin(t * 0.01f) * 50.0f, std::cos(t * 0.01f) * 50.0f, t * 0.1f,
            0.0f, 1.0f, 0.0f,
            std::fmod(t * 0.001f, 1.0f), std::fmod(t * 0.002f, 1.0f)
        };
        posBuffer[i * 3 + 0] = rawVertices[i].px;
        posBuffer[i * 3 + 1] = rawVertices[i].py;
        posBuffer[i * 3 + 2] = rawVertices[i].pz;
    }

    // Measure Highway SIMD Processing Speed
    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<QuantizedVertexHighQuality> qverts(testVertexCount);
    std::vector<uint16_t> quantizedPos(testVertexCount * 3);

    // Call vectorized Highway SIMD position quantization
    QuantizePositionsHighway(posBuffer.data(), quantizedPos.data(), posBuffer.size());

    for (size_t i = 0; i < testVertexCount; ++i) {
        qverts[i].posX = quantizedPos[i * 3 + 0];
        qverts[i].posY = quantizedPos[i * 3 + 1];
        qverts[i].posZ = quantizedPos[i * 3 + 2];
        qverts[i].unusedPadding = 0;
        qverts[i].normOctX = 0;
        qverts[i].normOctY = 32767;
        qverts[i].u = static_cast<uint16_t>(rawVertices[i].u * 65535.0f);
        qverts[i].v = static_cast<uint16_t>(rawVertices[i].v * 65535.0f);
    }

    Type0MeshHeader meshMeta;
    meshMeta.vertexCount = static_cast<uint32_t>(testVertexCount);
    meshMeta.vertexStride = sizeof(QuantizedVertexHighQuality);

    std::vector<uint8_t> packedPayload(sizeof(Type0MeshHeader) + qverts.size() * sizeof(QuantizedVertexHighQuality));
    std::memcpy(packedPayload.data(), &meshMeta, sizeof(meshMeta));
    std::memcpy(packedPayload.data() + sizeof(meshMeta), qverts.data(), qverts.size() * sizeof(QuantizedVertexHighQuality));

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    uint64_t payloadHash = HashBufferFNV1a(packedPayload.data(), packedPayload.size());

    std::cout << "[Google Highway SIMD] Vectorized Quantization Complete\n";
    std::cout << "  -> Vertex Count:     " << testVertexCount << "\n";
    std::cout << "  -> Processing Time:  " << elapsedMs << " ms\n";
    std::cout << "  -> Payload Size:     " << packedPayload.size() << " bytes (" << (packedPayload.size() / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "  -> Hermetic SIMD Hash: 0x" << std::hex << payloadHash << std::dec << "\n";
}

int main(int argc, char** argv) {
    RunHighwaySIMDCookerBenchmark();
    return 0;
}
