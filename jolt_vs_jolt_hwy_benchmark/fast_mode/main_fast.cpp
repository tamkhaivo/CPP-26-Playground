#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <thread>

// Jolt includes (Fast / Default Mode)
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

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

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    virtual const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const override {
        switch ((JPH::BroadPhaseLayer::Type)inLayer) {
        case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING: return "NON_MOVING";
        case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::MOVING: return "MOVING";
        default: return "INVALID";
        }
    }
#endif

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

uint64_t ComputeStateHash(const void* data, size_t size) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data);
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= ptr[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

int main() {
    std::cout << "===============================================================\n";
    std::cout << "          MODE 1: PURE JOLT (FAST / DEFAULT NON-DET)           \n";
    std::cout << "===============================================================\n";
    std::cout << "[Compiler Flag] Fast-Math /fp:fast: ENABLED\n";
    std::cout << "[Job System] ThreadPool Multi-Threaded Worker Pool\n";

    JPH::RegisterDefaultAllocator();
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();

    const size_t NUM_BODIES = 15000;
    const int NUM_FRAMES = 500;
    const float DELTA_TIME = 1.0f / 60.0f;

    BPLayerInterfaceImpl broad_phase_layer_interface;
    ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
    ObjectLayerPairFilterImpl object_vs_object_layer_filter;

    JPH::TempAllocatorImpl temp_allocator(32 * 1024 * 1024);
    JPH::JobSystemThreadPool job_system(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, std::thread::hardware_concurrency());

    JPH::PhysicsSystem physics_system;
    physics_system.Init(static_cast<uint32_t>(NUM_BODIES + 10), 0, 2048, 2048, broad_phase_layer_interface, object_vs_broadphase_layer_filter, object_vs_object_layer_filter);
    JPH::BodyInterface& body_interface = physics_system.GetBodyInterface();

    // Container floor & angled ramps
    JPH::BoxShapeSettings floor_settings(JPH::Vec3(200.0f, 1.0f, 200.0f));
    JPH::ShapeRefC floor_shape = floor_settings.Create().Get();
    JPH::BodyCreationSettings floor_bcs(floor_shape, JPH::RVec3(0.0f, -1.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::NON_MOVING);
    body_interface.CreateAndAddBody(floor_bcs, JPH::EActivation::DontActivate);

    // 15,000 dynamic bodies stacked in dense grid columns (spheres & boxes)
    JPH::SphereShapeSettings sphere_shape_settings(0.4f);
    JPH::ShapeRefC sphere_shape = sphere_shape_settings.Create().Get();
    JPH::BoxShapeSettings box_shape_settings(JPH::Vec3(0.4f, 0.4f, 0.4f));
    JPH::ShapeRefC box_shape = box_shape_settings.Create().Get();

    std::vector<JPH::BodyID> body_ids;
    body_ids.reserve(NUM_BODIES);

    for (size_t i = 0; i < NUM_BODIES; ++i) {
        float x = static_cast<float>(i % 50) - 25.0f;
        float y = 5.0f + static_cast<float>(i / (50 * 50)) * 1.1f;
        float z = static_cast<float>((i / 50) % 50) - 25.0f;

        JPH::ShapeRefC active_shape = (i % 2 == 0) ? sphere_shape : box_shape;
        JPH::BodyCreationSettings sphere_bcs(active_shape, JPH::RVec3(x, y, z), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
        sphere_bcs.mFriction = 0.5f;
        sphere_bcs.mRestitution = 0.3f;
        
        JPH::Body* body = body_interface.CreateBody(sphere_bcs);
        body_ids.push_back(body->GetID());
        body_interface.AddBody(body->GetID(), JPH::EActivation::Activate);
    }

    std::cout << "Simulating 15,000 dynamic bodies over 500 frames...\n";
    auto start = std::chrono::high_resolution_clock::now();
    for (int frame = 0; frame < NUM_FRAMES; ++frame) {
        physics_system.Update(DELTA_TIME, 1, &temp_allocator, &job_system);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    std::vector<float> final_positions;
    final_positions.reserve(NUM_BODIES * 3);
    for (auto id : body_ids) {
        JPH::RVec3 pos = body_interface.GetPosition(id);
        final_positions.push_back(static_cast<float>(pos.GetX()));
        final_positions.push_back(static_cast<float>(pos.GetY()));
        final_positions.push_back(static_cast<float>(pos.GetZ()));
    }
    uint64_t state_hash = ComputeStateHash(final_positions.data(), final_positions.size() * sizeof(float));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Total Execution Time: " << duration << " ms (" << duration / NUM_FRAMES << " ms/frame)\n";
    std::cout << "State Checksum Hash : 0x" << std::hex << std::uppercase << state_hash << std::dec << "\n";

    delete JPH::Factory::sInstance;
    return 0;
}
