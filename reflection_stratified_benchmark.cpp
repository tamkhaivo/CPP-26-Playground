#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <numeric>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <tuple>

// ============================================================================
// Data Structures: Vulkan/SIMD Aligned ECS Components
// ============================================================================

struct alignas(16) TransformComponent {
    float posX, posY, posZ, posW;
    float rotX, rotY, rotZ, rotW;
    float scaleX, scaleY, scaleZ, scaleW;
};

struct alignas(16) RigidBodyComponent {
    float velX, velY, velZ, mass;
    float angVelX, angVelY, angVelZ, restitution;
    uint32_t flags;
    uint32_t layerMask;
    uint32_t entityID;
    uint32_t padding;
};

struct alignas(16) ParticleComponent {
    float px, py, pz, lifetime;
    float vx, vy, vz, size;
    uint32_t colorRGBA;
    uint32_t emitterID;
    uint32_t active;
    uint32_t pad;
};

struct EntityData {
    TransformComponent transform;
    RigidBodyComponent body;
    ParticleComponent particle;
};

// ============================================================================
// Hashing Helper (FNV-1a 64-bit for Determinism Checks)
// ============================================================================

static uint64_t FNV1a_Hash(const uint8_t* data, size_t size) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

// ============================================================================
// Strategy A: RTTR-Style Dynamic Runtime Reflection Engine
// ============================================================================

namespace Strategy_RTTR {
    struct FieldMeta {
        std::string name;
        size_t offset;
        size_t size;
    };

    struct TypeMeta {
        std::string name;
        size_t totalSize;
        std::vector<FieldMeta> fields;
    };

    class RuntimeRegistry {
    public:
        static RuntimeRegistry& Instance() {
            static RuntimeRegistry instance;
            return instance;
        }

        void RegisterTransform() {
            TypeMeta meta{"TransformComponent", sizeof(TransformComponent), {
                {"posX", offsetof(TransformComponent, posX), sizeof(float)},
                {"posY", offsetof(TransformComponent, posY), sizeof(float)},
                {"posZ", offsetof(TransformComponent, posZ), sizeof(float)},
                {"posW", offsetof(TransformComponent, posW), sizeof(float)},
                {"rotX", offsetof(TransformComponent, rotX), sizeof(float)},
                {"rotY", offsetof(TransformComponent, rotY), sizeof(float)},
                {"rotZ", offsetof(TransformComponent, rotZ), sizeof(float)},
                {"rotW", offsetof(TransformComponent, rotW), sizeof(float)},
                {"scaleX", offsetof(TransformComponent, scaleX), sizeof(float)},
                {"scaleY", offsetof(TransformComponent, scaleY), sizeof(float)},
                {"scaleZ", offsetof(TransformComponent, scaleZ), sizeof(float)},
                {"scaleW", offsetof(TransformComponent, scaleW), sizeof(float)}
            }};
            types["TransformComponent"] = meta;
        }

        void RegisterRigidBody() {
            TypeMeta meta{"RigidBodyComponent", sizeof(RigidBodyComponent), {
                {"velX", offsetof(RigidBodyComponent, velX), sizeof(float)},
                {"velY", offsetof(RigidBodyComponent, velY), sizeof(float)},
                {"velZ", offsetof(RigidBodyComponent, velZ), sizeof(float)},
                {"mass", offsetof(RigidBodyComponent, mass), sizeof(float)},
                {"angVelX", offsetof(RigidBodyComponent, angVelX), sizeof(float)},
                {"angVelY", offsetof(RigidBodyComponent, angVelY), sizeof(float)},
                {"angVelZ", offsetof(RigidBodyComponent, angVelZ), sizeof(float)},
                {"restitution", offsetof(RigidBodyComponent, restitution), sizeof(float)},
                {"flags", offsetof(RigidBodyComponent, flags), sizeof(uint32_t)},
                {"layerMask", offsetof(RigidBodyComponent, layerMask), sizeof(uint32_t)},
                {"entityID", offsetof(RigidBodyComponent, entityID), sizeof(uint32_t)},
                {"padding", offsetof(RigidBodyComponent, padding), sizeof(uint32_t)}
            }};
            types["RigidBodyComponent"] = meta;
        }

        void RegisterParticle() {
            TypeMeta meta{"ParticleComponent", sizeof(ParticleComponent), {
                {"px", offsetof(ParticleComponent, px), sizeof(float)},
                {"py", offsetof(ParticleComponent, py), sizeof(float)},
                {"pz", offsetof(ParticleComponent, pz), sizeof(float)},
                {"lifetime", offsetof(ParticleComponent, lifetime), sizeof(float)},
                {"vx", offsetof(ParticleComponent, vx), sizeof(float)},
                {"vy", offsetof(ParticleComponent, vy), sizeof(float)},
                {"vz", offsetof(ParticleComponent, vz), sizeof(float)},
                {"size", offsetof(ParticleComponent, size), sizeof(float)},
                {"colorRGBA", offsetof(ParticleComponent, colorRGBA), sizeof(uint32_t)},
                {"emitterID", offsetof(ParticleComponent, emitterID), sizeof(uint32_t)},
                {"active", offsetof(ParticleComponent, active), sizeof(uint32_t)},
                {"pad", offsetof(ParticleComponent, pad), sizeof(uint32_t)}
            }};
            types["ParticleComponent"] = meta;
        }

        std::unordered_map<std::string, TypeMeta> types;
    };

    void Serialize(const std::vector<EntityData>& entities, std::vector<uint8_t>& buffer) {
        buffer.resize(entities.size() * sizeof(EntityData));
        auto& reg = RuntimeRegistry::Instance();
        auto& transformMeta = reg.types["TransformComponent"];
        auto& bodyMeta = reg.types["RigidBodyComponent"];
        auto& particleMeta = reg.types["ParticleComponent"];

        uint8_t* writePtr = buffer.data();

        for (const auto& e : entities) {
            const uint8_t* tBase = reinterpret_cast<const uint8_t*>(&e.transform);
            for (const auto& f : transformMeta.fields) {
                std::memcpy(writePtr, tBase + f.offset, f.size);
                writePtr += f.size;
            }
            const uint8_t* bBase = reinterpret_cast<const uint8_t*>(&e.body);
            for (const auto& f : bodyMeta.fields) {
                std::memcpy(writePtr, bBase + f.offset, f.size);
                writePtr += f.size;
            }
            const uint8_t* pBase = reinterpret_cast<const uint8_t*>(&e.particle);
            for (const auto& f : particleMeta.fields) {
                std::memcpy(writePtr, pBase + f.offset, f.size);
                writePtr += f.size;
            }
        }
    }

    void Deserialize(const std::vector<uint8_t>& buffer, std::vector<EntityData>& outEntities, size_t count) {
        outEntities.resize(count);
        auto& reg = RuntimeRegistry::Instance();
        auto& transformMeta = reg.types["TransformComponent"];
        auto& bodyMeta = reg.types["RigidBodyComponent"];
        auto& particleMeta = reg.types["ParticleComponent"];

        size_t readOffset = 0;
        for (size_t i = 0; i < count; ++i) {
            uint8_t* tBase = reinterpret_cast<uint8_t*>(&outEntities[i].transform);
            for (const auto& f : transformMeta.fields) {
                std::memcpy(tBase + f.offset, buffer.data() + readOffset, f.size);
                readOffset += f.size;
            }

            uint8_t* bBase = reinterpret_cast<uint8_t*>(&outEntities[i].body);
            for (const auto& f : bodyMeta.fields) {
                std::memcpy(bBase + f.offset, buffer.data() + readOffset, f.size);
                readOffset += f.size;
            }

            uint8_t* pBase = reinterpret_cast<uint8_t*>(&outEntities[i].particle);
            for (const auto& f : particleMeta.fields) {
                std::memcpy(pBase + f.offset, buffer.data() + readOffset, f.size);
                readOffset += f.size;
            }
        }
    }
}

// ============================================================================
// Strategy B: `refl-cpp` Style Constexpr Reflection
// ============================================================================

namespace Strategy_ReflCpp {
    template<typename Tuple, typename Func, size_t... Is>
    constexpr void for_each_impl(Tuple&& t, Func&& f, std::index_sequence<Is...>) {
        (f(std::get<Is>(t)), ...);
    }

    template<typename... Args, typename Func>
    constexpr void reflect_for_each(std::tuple<Args...>&& t, Func&& f) {
        for_each_impl(std::forward<std::tuple<Args...>>(t), std::forward<Func>(f), std::index_sequence_for<Args...>{});
    }

    constexpr auto ReflectTransform(const TransformComponent& t) {
        return std::tie(t.posX, t.posY, t.posZ, t.posW, t.rotX, t.rotY, t.rotZ, t.rotW, t.scaleX, t.scaleY, t.scaleZ, t.scaleW);
    }

    constexpr auto ReflectTransformMut(TransformComponent& t) {
        return std::tie(t.posX, t.posY, t.posZ, t.posW, t.rotX, t.rotY, t.rotZ, t.rotW, t.scaleX, t.scaleY, t.scaleZ, t.scaleW);
    }

    constexpr auto ReflectRigidBody(const RigidBodyComponent& b) {
        return std::tie(b.velX, b.velY, b.velZ, b.mass, b.angVelX, b.angVelY, b.angVelZ, b.restitution, b.flags, b.layerMask, b.entityID, b.padding);
    }

    constexpr auto ReflectRigidBodyMut(RigidBodyComponent& b) {
        return std::tie(b.velX, b.velY, b.velZ, b.mass, b.angVelX, b.angVelY, b.angVelZ, b.restitution, b.flags, b.layerMask, b.entityID, b.padding);
    }

    constexpr auto ReflectParticle(const ParticleComponent& p) {
        return std::tie(p.px, p.py, p.pz, p.lifetime, p.vx, p.vy, p.vz, p.size, p.colorRGBA, p.emitterID, p.active, p.pad);
    }

    constexpr auto ReflectParticleMut(ParticleComponent& p) {
        return std::tie(p.px, p.py, p.pz, p.lifetime, p.vx, p.vy, p.vz, p.size, p.colorRGBA, p.emitterID, p.active, p.pad);
    }

    void Serialize(const std::vector<EntityData>& entities, std::vector<uint8_t>& buffer) {
        buffer.resize(entities.size() * sizeof(EntityData));
        uint8_t* writePtr = buffer.data();

        for (const auto& e : entities) {
            reflect_for_each(ReflectTransform(e.transform), [&](const auto& field) {
                using T = std::decay_t<decltype(field)>;
                std::memcpy(writePtr, &field, sizeof(T));
                writePtr += sizeof(T);
            });
            reflect_for_each(ReflectRigidBody(e.body), [&](const auto& field) {
                using T = std::decay_t<decltype(field)>;
                std::memcpy(writePtr, &field, sizeof(T));
                writePtr += sizeof(T);
            });
            reflect_for_each(ReflectParticle(e.particle), [&](const auto& field) {
                using T = std::decay_t<decltype(field)>;
                std::memcpy(writePtr, &field, sizeof(T));
                writePtr += sizeof(T);
            });
        }
    }

    void Deserialize(const std::vector<uint8_t>& buffer, std::vector<EntityData>& outEntities, size_t count) {
        outEntities.resize(count);
        size_t readOffset = 0;

        for (size_t i = 0; i < count; ++i) {
            reflect_for_each(ReflectTransformMut(outEntities[i].transform), [&](auto& field) {
                using T = std::decay_t<decltype(field)>;
                std::memcpy(&field, buffer.data() + readOffset, sizeof(T));
                readOffset += sizeof(T);
            });
            reflect_for_each(ReflectRigidBodyMut(outEntities[i].body), [&](auto& field) {
                using T = std::decay_t<decltype(field)>;
                std::memcpy(&field, buffer.data() + readOffset, sizeof(T));
                readOffset += sizeof(T);
            });
            reflect_for_each(ReflectParticleMut(outEntities[i].particle), [&](auto& field) {
                using T = std::decay_t<decltype(field)>;
                std::memcpy(&field, buffer.data() + readOffset, sizeof(T));
                readOffset += sizeof(T);
            });
        }
    }
}

// ============================================================================
// Strategy C: `bitsery` Style Bit-Packed / Zero-Copy Memory Serializer
// ============================================================================

namespace Strategy_Bitsery {
    void Serialize(const std::vector<EntityData>& entities, std::vector<uint8_t>& buffer) {
        const size_t totalBytes = entities.size() * sizeof(EntityData);
        buffer.resize(totalBytes);
        std::memcpy(buffer.data(), entities.data(), totalBytes);
    }

    void Deserialize(const std::vector<uint8_t>& buffer, std::vector<EntityData>& outEntities, size_t count) {
        outEntities.resize(count);
        std::memcpy(outEntities.data(), buffer.data(), count * sizeof(EntityData));
    }
}

// ============================================================================
// Strategy D: `cereal` Style Stream Archive Serializer
// ============================================================================

namespace Strategy_Cereal {
    void Serialize(const std::vector<EntityData>& entities, std::vector<uint8_t>& buffer) {
        const size_t payloadSize = 4 + entities.size() * (12 + sizeof(EntityData));
        buffer.resize(payloadSize);
        uint8_t* ptr = buffer.data();

        uint32_t count = static_cast<uint32_t>(entities.size());
        std::memcpy(ptr, &count, sizeof(count));
        ptr += sizeof(count);

        for (const auto& e : entities) {
            std::memcpy(ptr, "TRSF", 4); ptr += 4;
            std::memcpy(ptr, &e.transform, sizeof(TransformComponent)); ptr += sizeof(TransformComponent);

            std::memcpy(ptr, "BODY", 4); ptr += 4;
            std::memcpy(ptr, &e.body, sizeof(RigidBodyComponent)); ptr += sizeof(RigidBodyComponent);

            std::memcpy(ptr, "PART", 4); ptr += 4;
            std::memcpy(ptr, &e.particle, sizeof(ParticleComponent)); ptr += sizeof(ParticleComponent);
        }
    }

    void Deserialize(const std::vector<uint8_t>& buffer, std::vector<EntityData>& outEntities) {
        uint32_t count = 0;
        std::memcpy(&count, buffer.data(), sizeof(count));
        outEntities.resize(count);

        const uint8_t* ptr = buffer.data() + sizeof(count);
        for (size_t i = 0; i < count; ++i) {
            ptr += 4; // tag TRSF
            std::memcpy(&outEntities[i].transform, ptr, sizeof(TransformComponent)); ptr += sizeof(TransformComponent);
            ptr += 4; // tag BODY
            std::memcpy(&outEntities[i].body, ptr, sizeof(RigidBodyComponent)); ptr += sizeof(RigidBodyComponent);
            ptr += 4; // tag PART
            std::memcpy(&outEntities[i].particle, ptr, sizeof(ParticleComponent)); ptr += sizeof(ParticleComponent);
        }
    }
}

// ============================================================================
// Strategy E: Type0 Hybrid Engine Architecture (`refl-cpp` GUI + `bitsery` Persistence)
// ============================================================================

namespace Strategy_HybridType0 {
    double MeasurePropertyInspection(std::vector<EntityData>& entities) {
        auto start = std::chrono::high_resolution_clock::now();
        float dummySum = 0.0f;
        for (auto& e : entities) {
            Strategy_ReflCpp::reflect_for_each(Strategy_ReflCpp::ReflectTransform(e.transform), [&](const auto& field) {
                dummySum += static_cast<float>(field);
            });
        }
        auto end = std::chrono::high_resolution_clock::now();
        if (dummySum == 1234567.0f) std::cout << " ";
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    void Serialize(const std::vector<EntityData>& entities, std::vector<uint8_t>& buffer) {
        Strategy_Bitsery::Serialize(entities, buffer);
    }

    void Deserialize(const std::vector<uint8_t>& buffer, std::vector<EntityData>& outEntities, size_t count) {
        Strategy_Bitsery::Deserialize(buffer, outEntities, count);
    }
}

// ============================================================================
// Main Execution & Multi-Run Rigorous Benchmark
// ============================================================================

int main() {
    std::cout << "==========================================================================\n";
    std::cout << "  RIGOROUS RE-VERIFICATION BENCHMARK (100 ITERATIONS WITH MIN/MED/MAX)\n";
    std::cout << "  Target: 100,000 ECS Entities (Aligned struct layout 144 bytes / entity)\n";
    std::cout << "==========================================================================\n\n";

    // Register RTTR
    Strategy_RTTR::RuntimeRegistry::Instance().RegisterTransform();
    Strategy_RTTR::RuntimeRegistry::Instance().RegisterRigidBody();
    Strategy_RTTR::RuntimeRegistry::Instance().RegisterParticle();

    const size_t entityCount = 100000;
    std::vector<EntityData> entities(entityCount);

    for (size_t i = 0; i < entityCount; ++i) {
        entities[i].transform = { (float)i, (float)i * 1.1f, (float)i * 1.2f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
        entities[i].body = { 0.1f * i, 0.2f * i, 0.3f * i, 75.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0x1, 0xFF, (uint32_t)i, 0 };
        entities[i].particle = { 1.0f * i, 2.0f * i, 3.0f * i, 5.0f, 0.1f, 0.2f, 0.3f, 0.5f, 0xFFFFFFFF, 12, 1, 0 };
    }

    const double uncompressedDataMB = (entityCount * sizeof(EntityData)) / (1024.0 * 1024.0);
    const int iterations = 100;

    struct MultiRunResult {
        std::string name;
        double minSerMs, medianSerMs, maxSerMs;
        double minDeserMs, medianDeserMs, maxDeserMs;
        double medianSerThroughputMBs;
        double medianDeserThroughputMBs;
        size_t payloadSizeBytes;
        uint64_t hash;
        bool deterministic;
    };

    std::vector<MultiRunResult> results;

    auto BenchmarkStrategyMulti = [&](const std::string& name, auto serFunc, auto deserFunc) {
        std::vector<uint8_t> buffer;
        std::vector<EntityData> deserializedEntities;

        // Warmup run
        serFunc(entities, buffer);
        deserFunc(buffer, deserializedEntities, entityCount);

        std::vector<double> serTimes;
        std::vector<double> deserTimes;
        serTimes.reserve(iterations);
        deserTimes.reserve(iterations);

        uint64_t baseHash = 0;
        bool deterministic = true;

        for (int it = 0; it < iterations; ++it) {
            buffer.clear();
            auto startSer = std::chrono::high_resolution_clock::now();
            serFunc(entities, buffer);
            auto endSer = std::chrono::high_resolution_clock::now();
            serTimes.push_back(std::chrono::duration<double, std::milli>(endSer - startSer).count());

            auto startDeser = std::chrono::high_resolution_clock::now();
            deserFunc(buffer, deserializedEntities, entityCount);
            auto endDeser = std::chrono::high_resolution_clock::now();
            deserTimes.push_back(std::chrono::duration<double, std::milli>(endDeser - startDeser).count());

            uint64_t h = FNV1a_Hash(buffer.data(), buffer.size());
            if (it == 0) baseHash = h;
            else if (h != baseHash) deterministic = false;
        }

        std::sort(serTimes.begin(), serTimes.end());
        std::sort(deserTimes.begin(), deserTimes.end());

        double medSer = serTimes[iterations / 2];
        double medDeser = deserTimes[iterations / 2];

        results.push_back({
            name,
            serTimes.front(), medSer, serTimes.back(),
            deserTimes.front(), medDeser, deserTimes.back(),
            uncompressedDataMB / (medSer / 1000.0),
            uncompressedDataMB / (medDeser / 1000.0),
            buffer.size(),
            baseHash,
            deterministic
        });
    };

    BenchmarkStrategyMulti("A: RTTR Dynamic Map", Strategy_RTTR::Serialize, Strategy_RTTR::Deserialize);
    BenchmarkStrategyMulti("B: refl-cpp Constexpr", Strategy_ReflCpp::Serialize, Strategy_ReflCpp::Deserialize);
    BenchmarkStrategyMulti("C: bitsery Bit-Packed", Strategy_Bitsery::Serialize, Strategy_Bitsery::Deserialize);
    BenchmarkStrategyMulti("D: cereal Stream Archive", Strategy_Cereal::Serialize, [](const std::vector<uint8_t>& buf, std::vector<EntityData>& out, size_t count) {
        Strategy_Cereal::Deserialize(buf, out);
    });
    BenchmarkStrategyMulti("E: Type0 Hybrid Pipeline", Strategy_HybridType0::Serialize, Strategy_HybridType0::Deserialize);

    // Print Results Table
    std::cout << std::left 
              << std::setw(28) << "Candidate Strategy"
              << std::setw(22) << "Ser Med (Min/Max) ms"
              << std::setw(18) << "Ser Speed (MB/s)"
              << std::setw(22) << "Deser Med (Min/Max)"
              << std::setw(18) << "Deser Speed(MB/s)"
              << std::setw(14) << "Deterministic"
              << "\n";
    std::cout << std::string(122, '-') << "\n";

    for (const auto& r : results) {
        std::stringstream ssSer, ssDeser;
        ssSer << std::fixed << std::setprecision(2) << r.medianSerMs << " (" << r.minSerMs << "/" << r.maxSerMs << ")";
        ssDeser << std::fixed << std::setprecision(2) << r.medianDeserMs << " (" << r.minDeserMs << "/" << r.maxDeserMs << ")";

        std::cout << std::left
                  << std::setw(28) << r.name
                  << std::setw(22) << ssSer.str()
                  << std::setw(18) << std::fixed << std::setprecision(1) << r.medianSerThroughputMBs
                  << std::setw(22) << ssDeser.str()
                  << std::setw(18) << std::fixed << std::setprecision(1) << r.medianDeserThroughputMBs
                  << std::setw(14) << (r.deterministic ? "YES (PASSED)" : "NO (FAILED)")
                  << "\n";
    }

    std::cout << std::string(122, '-') << "\n\n";
    return 0;
}
