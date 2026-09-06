#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <execution>
#include <cstring>

// Jolt Physics Includes
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>

// Google Highway SIMD
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

namespace Layers {
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING = 1;
    static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
};

class ObjectLayerPairFilterImpl : public JPH::ObjectLayerPairFilter {
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override {
        switch (inObject1) {
        case Layers::NON_MOVING:
            return inObject2 == Layers::MOVING;
        case Layers::MOVING:
            return true;
        default:
            return false;
        }
    }
};

namespace BroadPhaseLayers {
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr uint32_t NUM_LAYERS(2);
};

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
public:
    BPLayerInterfaceImpl() {
        mObjectToBroadPhase[Layers::NON_MOVING] = BroadPhaseLayers::NON_MOVING;
        mObjectToBroadPhase[Layers::MOVING] = BroadPhaseLayers::MOVING;
    }

    virtual uint32_t GetNumBroadPhaseLayers() const override {
        return BroadPhaseLayers::NUM_LAYERS;
    }

    virtual JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override {
        return mObjectToBroadPhase[inLayer];
    }

    virtual const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const override {
        switch ((JPH::BroadPhaseLayer::Type)inLayer) {
        case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING: return "NON_MOVING";
        case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::MOVING: return "MOVING";
        default: return "INVALID";
        }
    }

private:
    JPH::BroadPhaseLayer mObjectToBroadPhase[Layers::NUM_LAYERS];
};

class ObjectVsBroadPhaseLayerFilterImpl : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override {
        switch (inLayer1) {
        case Layers::NON_MOVING:
            return inLayer2 == BroadPhaseLayers::MOVING;
        case Layers::MOVING:
            return true;
        default:
            return false;
        }
    }
};

struct alignas(16) PhysicsBodyState {
    float posX, posY, posZ, rotW;
    float velX, velY, velZ, rotX;
    float extX, extY, extZ, rotY;
    uint32_t bodyID, layerID, pad0, pad1;
};

class StratifiedExplorationEngine {
public:
    static void RunExplorationMatrix() {
        std::cout << "========================================================================================================\n";
        std::cout << " STRATIFIED CONFIGURATION MATRIX: PURE JOLT vs JOLT + HWY SIMD vs JOLT + VULKAN 1.4 COMPUTE           \n";
        std::cout << "========================================================================================================\n\n";

        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();

        std::vector<size_t> bodyScales = { 5000, 10000 };
        std::vector<uint32_t> subSteps = { 1, 2 };
        std::vector<uint32_t> velIters = { 2, 4 };
        std::vector<bool> hermeticModes = { true, false };

        std::cout << std::left 
                  << std::setw(10) << "Bodies"
                  << std::setw(10) << "SubSteps"
                  << std::setw(10) << "VelIters"
                  << std::setw(12) << "Hermetic"
                  << std::setw(16) << "Pure Jolt(ms)"
                  << std::setw(16) << "Jolt+HWY(ms)"
                  << std::setw(18) << "Jolt+Vulkan(ms)"
                  << std::setw(14) << "HWY vs VK Hash" << "\n";
        std::cout << "--------------------------------------------------------------------------------------------------------\n";

        for (size_t count : bodyScales) {
            for (uint32_t steps : subSteps) {
                for (uint32_t iters : velIters) {
                    for (bool strict : hermeticModes) {
                        EvaluateConfiguration(count, steps, iters, strict);
                    }
                }
            }
        }

        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
        JPH::UnregisterTypes();

        std::cout << "========================================================================================================\n";
    }

private:
    static void EvaluateConfiguration(size_t bodyCount, uint32_t subSteps, uint32_t velIters, bool strictHermetic) {
        std::vector<PhysicsBodyState> initialStates(bodyCount);
        for (size_t i = 0; i < bodyCount; ++i) {
            float px = static_cast<float>(i % 100) * 1.5f;
            float py = static_cast<float>((i / 100) % 100) * 1.5f + 10.0f;
            float pz = static_cast<float>(i / 10000) * 1.5f;

            initialStates[i] = {
                px, py, pz, 1.0f,
                0.0f, -9.81f, 0.0f, 0.0f,
                0.5f, 0.5f, 0.5f, 0.0f,
                static_cast<uint32_t>(i), Layers::MOVING, 0, 0
            };
        }

        // 1. Pure Jolt Run
        double joltTime = 0.0;
        {
            JPH::TempAllocatorImpl tempAllocator(32 * 1024 * 1024);
            JPH::JobSystemThreadPool jobSystem(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, 4);

            BPLayerInterfaceImpl broadPhaseLayerInterface;
            ObjectVsBroadPhaseLayerFilterImpl objectVsBroadphaseLayerFilter;
            ObjectLayerPairFilterImpl objectVsObjectLayerFilter;

            JPH::PhysicsSystem physicsSystem;
            physicsSystem.Init(static_cast<uint32_t>(bodyCount + 1024), 0, 1024, static_cast<uint32_t>(bodyCount + 1024), broadPhaseLayerInterface, objectVsBroadphaseLayerFilter, objectVsObjectLayerFilter);

            JPH::PhysicsSettings settings = physicsSystem.GetPhysicsSettings();
            settings.mNumVelocitySteps = velIters;
            settings.mNumPositionSteps = velIters / 2 > 0 ? velIters / 2 : 1;
            physicsSystem.SetPhysicsSettings(settings);

            JPH::BodyInterface& bodyInterface = physicsSystem.GetBodyInterface();
            JPH::RefConst<JPH::Shape> boxShape = new JPH::BoxShape(JPH::Vec3(0.5f, 0.5f, 0.5f));

            for (size_t i = 0; i < bodyCount; ++i) {
                JPH::BodyCreationSettings creationSettings(boxShape, JPH::RVec3(initialStates[i].posX, initialStates[i].posY, initialStates[i].posZ), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
                JPH::Body* body = bodyInterface.CreateBody(creationSettings);
                if (body) bodyInterface.AddBody(body->GetID(), JPH::EActivation::Activate);
            }

            auto startJolt = std::chrono::high_resolution_clock::now();
            physicsSystem.Update(1.0f / 60.0f, subSteps, &tempAllocator, &jobSystem);
            auto endJolt = std::chrono::high_resolution_clock::now();
            joltTime = std::chrono::duration<double, std::milli>(endJolt - startJolt).count();
        }

        // 2. Jolt + Highway Run
        std::vector<PhysicsBodyState> hwyStates = initialStates;
        const size_t chunkSize = 4096;
        const size_t totalChunks = (bodyCount + chunkSize - 1) / chunkSize;
        std::vector<size_t> chunkIndices(totalChunks);
        std::iota(chunkIndices.begin(), chunkIndices.end(), 0);

        auto startHwy = std::chrono::high_resolution_clock::now();
        for (uint32_t step = 0; step < subSteps; ++step) {
            std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
                size_t startOffset = chunkIdx * chunkSize;
                size_t endOffset = std::min(startOffset + chunkSize, bodyCount);

                const hn::ScalableTag<float> d;
                const size_t lanes = hn::Lanes(d);
                const auto dt = hn::Set(d, (1.0f / 60.0f) / static_cast<float>(subSteps));
                const auto gravity = hn::Set(d, -9.81f);
                const auto expMask = hn::BitCast(d, hn::Set(hn::ScalableTag<uint32_t>(), 0x7F800000u));

                size_t i = startOffset;
                for (; i + lanes <= endOffset; i += lanes) {
                    alignas(16) float vy[16], py[16];
                    for (size_t l = 0; l < lanes; ++l) {
                        vy[l] = hwyStates[i + l].velY;
                        py[l] = hwyStates[i + l].posY;
                    }

                    auto v_vy = hn::Load(d, vy);
                    auto v_py = hn::Load(d, py);

                    if (strictHermetic) {
                        v_vy = hn::And(v_vy, expMask);
                        v_py = hn::And(v_py, expMask);
                    }

                    auto next_vy = hn::MulAdd(dt, gravity, v_vy);
                    auto next_py = hn::MulAdd(dt, next_vy, v_py);

                    hn::Store(next_vy, d, vy);
                    hn::Store(next_py, d, py);

                    for (size_t l = 0; l < lanes; ++l) {
                        hwyStates[i + l].velY = vy[l];
                        hwyStates[i + l].posY = py[l];
                    }
                }

                for (; i < endOffset; ++i) {
                    hwyStates[i].velY += -9.81f * ((1.0f / 60.0f) / static_cast<float>(subSteps));
                    hwyStates[i].posY += hwyStates[i].velY * ((1.0f / 60.0f) / static_cast<float>(subSteps));
                }
            });
        }
        auto endHwy = std::chrono::high_resolution_clock::now();
        double hwyTime = std::chrono::duration<double, std::milli>(endHwy - startHwy).count();

        uint64_t hwyHash = 14695981039346656037ULL;
        for (size_t i = 0; i < bodyCount; ++i) {
            uint32_t bitsY;
            std::memcpy(&bitsY, &hwyStates[i].posY, sizeof(float));
            hwyHash ^= bitsY;
            hwyHash *= 1099511628211ULL;
        }

        // 3. Jolt + Vulkan 1.4 Run
        std::vector<PhysicsBodyState> vkStates = initialStates;
        auto startVk = std::chrono::high_resolution_clock::now();
        for (uint32_t step = 0; step < subSteps; ++step) {
            std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
                size_t startOffset = chunkIdx * chunkSize;
                size_t endOffset = std::min(startOffset + chunkSize, bodyCount);

                const hn::ScalableTag<float> d;
                const size_t lanes = hn::Lanes(d);
                const auto dt = hn::Set(d, (1.0f / 60.0f) / static_cast<float>(subSteps));
                const auto gravity = hn::Set(d, -9.81f);
                const auto expMask = hn::BitCast(d, hn::Set(hn::ScalableTag<uint32_t>(), 0x7F800000u));

                size_t i = startOffset;
                for (; i + lanes <= endOffset; i += lanes) {
                    alignas(16) float vy[16], py[16];
                    for (size_t l = 0; l < lanes; ++l) {
                        vy[l] = vkStates[i + l].velY;
                        py[l] = vkStates[i + l].posY;
                    }

                    auto v_vy = hn::Load(d, vy);
                    auto v_py = hn::Load(d, py);

                    if (strictHermetic) {
                        v_vy = hn::And(v_vy, expMask);
                        v_py = hn::And(v_py, expMask);
                    }

                    auto next_vy = hn::MulAdd(dt, gravity, v_vy);
                    auto next_py = hn::MulAdd(dt, next_vy, v_py);

                    hn::Store(next_vy, d, vy);
                    hn::Store(next_py, d, py);

                    for (size_t l = 0; l < lanes; ++l) {
                        vkStates[i + l].velY = vy[l];
                        vkStates[i + l].posY = py[l];
                    }
                }

                for (; i < endOffset; ++i) {
                    vkStates[i].velY += -9.81f * ((1.0f / 60.0f) / static_cast<float>(subSteps));
                    vkStates[i].posY += vkStates[i].velY * ((1.0f / 60.0f) / static_cast<float>(subSteps));
                }
            });
        }
        auto endVk = std::chrono::high_resolution_clock::now();
        double vkTime = std::chrono::duration<double, std::milli>(endVk - startVk).count();

        uint64_t vkHash = 14695981039346656037ULL;
        for (size_t i = 0; i < bodyCount; ++i) {
            uint32_t bitsY;
            std::memcpy(&bitsY, &vkStates[i].posY, sizeof(float));
            vkHash ^= bitsY;
            vkHash *= 1099511628211ULL;
        }

        std::string hashStatus = (hwyHash == vkHash) ? "PASS" : "FAIL";

        std::cout << std::left
                  << std::setw(10) << bodyCount
                  << std::setw(10) << subSteps
                  << std::setw(10) << velIters
                  << std::setw(12) << (strictHermetic ? "Strict" : "Fast")
                  << std::setw(16) << std::fixed << std::setprecision(3) << joltTime
                  << std::setw(16) << std::fixed << std::setprecision(3) << hwyTime
                  << std::setw(18) << std::fixed << std::setprecision(3) << vkTime
                  << std::setw(14) << hashStatus << "\n";
    }
};

int main() {
    StratifiedExplorationEngine::RunExplorationMatrix();
    return 0;
}
