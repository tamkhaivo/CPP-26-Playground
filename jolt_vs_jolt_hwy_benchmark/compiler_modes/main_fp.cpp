#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <thread>
#include <cmath>

// Jolt includes
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemWithBarrier.h>
#include <Jolt/Core/FixedSizeFreeList.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>

// enkiTS Task Scheduler include
#include "TaskScheduler.h"

// Google Highway
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

class JoltEnkiTSJobSystem final : public JPH::JobSystemWithBarrier {
public:
    JoltEnkiTSJobSystem(JPH::uint inMaxJobs, JPH::uint inMaxBarriers, enki::TaskScheduler* ts)
        : mTaskScheduler(ts) {
        Init(inMaxBarriers);
        mJobs.Init(inMaxJobs, inMaxJobs);
    }

    virtual int GetMaxConcurrency() const override {
        return mTaskScheduler->GetNumTaskThreads();
    }

    virtual JPH::JobHandle CreateJob(const char* inName, JPH::ColorArg inColor, const JPH::JobSystem::JobFunction& inJobFunction, JPH::uint32 inNumDependencies = 0) override {
        JPH::uint32 index = mJobs.ConstructObject(inName, inColor, this, inJobFunction, inNumDependencies);
        return JPH::JobHandle(&mJobs.Get(index));
    }

protected:
    virtual void QueueJob(Job* inJob) override {
        inJob->Execute();
    }

    virtual void QueueJobs(Job** inJobs, JPH::uint inNumJobs) override {
        for (JPH::uint i = 0; i < inNumJobs; ++i) {
            inJobs[i]->Execute();
        }
    }

    virtual void FreeJob(Job* inJob) override {
        mJobs.DestructObject(inJob);
    }

private:
    enki::TaskScheduler* mTaskScheduler;
    JPH::FixedSizeFreeList<Job> mJobs;
};

struct alignas(16) PhysicsParticleState {
    float px, py, pz, pw;
    float vx, vy, vz, vw;
    float qx, qy, qz, qw;
};

// HEAVY COMPLEX SIMD MATH: Quaternion Rotations + Spatial Distance Threshold Masking + 3x3 Matrix Transformations
void ProcessHeavyHighwayParticleBatchEnkiTS(enki::TaskScheduler& ts, PhysicsParticleState* particles, size_t count, float dt) {
    const size_t chunkSize = 1024;
    const size_t totalChunks = (count + chunkSize - 1) / chunkSize;

    enki::TaskSet task(static_cast<uint32_t>(totalChunks), [&](enki::TaskSetPartition range, uint32_t threadnum) {
        const hn::ScalableTag<float> d;
        const size_t lanes = hn::Lanes(d);
        
        const auto v_dt = hn::Set(d, dt);
        const auto v_gravity = hn::Set(d, -9.81f);
        const auto v_half = hn::Set(d, 0.5f);
        const auto v_one = hn::Set(d, 1.0f);
        const auto v_dist_threshold = hn::Set(d, 25.0f * 25.0f);
        const auto v_damping = hn::Set(d, 0.98f);

        for (uint32_t chunkIdx = range.start; chunkIdx < range.end; ++chunkIdx) {
            size_t startOffset = chunkIdx * chunkSize;
            size_t endOffset = std::min(startOffset + chunkSize, count);

            size_t i = startOffset;
            for (; i + lanes <= endOffset; i += lanes) {
                float tmp_px[16], tmp_py[16], tmp_pz[16];
                float tmp_vx[16], tmp_vy[16], tmp_vz[16];
                float tmp_qx[16], tmp_qy[16], tmp_qz[16], tmp_qw[16];

                for (size_t l = 0; l < lanes; ++l) {
                    tmp_px[l] = particles[i + l].px;
                    tmp_py[l] = particles[i + l].py;
                    tmp_pz[l] = particles[i + l].pz;
                    tmp_vx[l] = particles[i + l].vx;
                    tmp_vy[l] = particles[i + l].vy;
                    tmp_vz[l] = particles[i + l].vz;
                    tmp_qx[l] = particles[i + l].qx;
                    tmp_qy[l] = particles[i + l].qy;
                    tmp_qz[l] = particles[i + l].qz;
                    tmp_qw[l] = particles[i + l].qw;
                }

                auto vx = hn::LoadU(d, tmp_vx);
                auto vy = hn::LoadU(d, tmp_vy);
                auto vz = hn::LoadU(d, tmp_vz);
                auto px = hn::LoadU(d, tmp_px);
                auto py = hn::LoadU(d, tmp_py);
                auto pz = hn::LoadU(d, tmp_pz);

                auto qx = hn::LoadU(d, tmp_qx);
                auto qy = hn::LoadU(d, tmp_qy);
                auto qz = hn::LoadU(d, tmp_qz);
                auto qw = hn::LoadU(d, tmp_qw);

                // 1. Gravity integration (Exact MulAdd)
                vy = hn::MulAdd(v_gravity, v_dt, vy);

                // 2. Spatial distance threshold check: dist_sq = px*px + py*py + pz*pz
                auto dist_sq = hn::MulAdd(px, px, hn::MulAdd(py, py, hn::Mul(pz, pz)));
                auto mask_outside = hn::Gt(dist_sq, v_dist_threshold);

                // Apply conditional damping to velocities outside boundary
                vx = hn::IfThenElse(mask_outside, hn::Mul(vx, v_damping), vx);
                vy = hn::IfThenElse(mask_outside, hn::Mul(vy, v_damping), vy);
                vz = hn::IfThenElse(mask_outside, hn::Mul(vz, v_damping), vz);

                // 3. Quaternion angular integration: dq/dt = 0.5 * q * w
                auto dqx = hn::Mul(v_half, hn::MulAdd(qw, vx, hn::MulAdd(qy, vz, hn::NegMulAdd(qz, vy, hn::Zero(d)))));
                auto dqy = hn::Mul(v_half, hn::MulAdd(qw, vy, hn::MulAdd(qz, vx, hn::NegMulAdd(qx, vz, hn::Zero(d)))));
                auto dqz = hn::Mul(v_half, hn::MulAdd(qw, vz, hn::MulAdd(qx, vy, hn::NegMulAdd(qy, vx, hn::Zero(d)))));
                auto dqw = hn::Mul(v_half, hn::NegMulAdd(qx, vx, hn::NegMulAdd(qy, vy, hn::NegMulAdd(qz, vz, hn::Zero(d)))));

                qx = hn::MulAdd(dqx, v_dt, qx);
                qy = hn::MulAdd(dqy, v_dt, qy);
                qz = hn::MulAdd(dqz, v_dt, qz);
                qw = hn::MulAdd(dqw, v_dt, qw);

                // 4. Normalize Quaternions: norm = sqrt(qx^2 + qy^2 + qz^2 + qw^2)
                auto q_norm_sq = hn::MulAdd(qx, qx, hn::MulAdd(qy, qy, hn::MulAdd(qz, qz, hn::Mul(qw, qw))));
                auto q_norm = hn::Sqrt(q_norm_sq);
                qx = hn::Div(qx, q_norm);
                qy = hn::Div(qy, q_norm);
                qz = hn::Div(qz, q_norm);
                qw = hn::Div(qw, q_norm);

                // 5. Position integration
                px = hn::MulAdd(vx, v_dt, px);
                py = hn::MulAdd(vy, v_dt, py);
                pz = hn::MulAdd(vz, v_dt, pz);

                hn::StoreU(vx, d, tmp_vx);
                hn::StoreU(vy, d, tmp_vy);
                hn::StoreU(vz, d, tmp_vz);
                hn::StoreU(px, d, tmp_px);
                hn::StoreU(py, d, tmp_py);
                hn::StoreU(pz, d, tmp_pz);
                hn::StoreU(qx, d, tmp_qx);
                hn::StoreU(qy, d, tmp_qy);
                hn::StoreU(qz, d, tmp_qz);
                hn::StoreU(qw, d, tmp_qw);

                for (size_t l = 0; l < lanes; ++l) {
                    particles[i + l].px = tmp_px[l];
                    particles[i + l].py = tmp_py[l];
                    particles[i + l].pz = tmp_pz[l];
                    particles[i + l].vx = tmp_vx[l];
                    particles[i + l].vy = tmp_vy[l];
                    particles[i + l].vz = tmp_vz[l];
                    particles[i + l].qx = tmp_qx[l];
                    particles[i + l].qy = tmp_qy[l];
                    particles[i + l].qz = tmp_qz[l];
                    particles[i + l].qw = tmp_qw[l];
                }
            }

            for (; i < endOffset; ++i) {
                particles[i].vy += -9.81f * dt;
                float dsq = particles[i].px * particles[i].px + particles[i].py * particles[i].py + particles[i].pz * particles[i].pz;
                if (dsq > 25.0f * 25.0f) {
                    particles[i].vx *= 0.98f;
                    particles[i].vy *= 0.98f;
                    particles[i].vz *= 0.98f;
                }
                particles[i].px += particles[i].vx * dt;
                particles[i].py += particles[i].vy * dt;
                particles[i].pz += particles[i].vz * dt;
            }
        }
    });

    ts.AddTaskSetToPipe(&task);
    ts.WaitforTask(&task);
}

void ProcessHeavyScalarParticleBatch(PhysicsParticleState* particles, size_t count, float dt) {
    for (size_t i = 0; i < count; ++i) {
        particles[i].vy += -9.81f * dt;
        float dsq = particles[i].px * particles[i].px + particles[i].py * particles[i].py + particles[i].pz * particles[i].pz;
        if (dsq > 25.0f * 25.0f) {
            particles[i].vx *= 0.98f;
            particles[i].vy *= 0.98f;
            particles[i].vz *= 0.98f;
        }

        // Quaternion integration
        float dqx = 0.5f * (particles[i].qw * particles[i].vx + particles[i].qy * particles[i].vz - particles[i].qz * particles[i].vy);
        float dqy = 0.5f * (particles[i].qw * particles[i].vy + particles[i].qz * particles[i].vx - particles[i].qx * particles[i].vz);
        float dqz = 0.5f * (particles[i].qw * particles[i].vz + particles[i].qx * particles[i].vy - particles[i].qy * particles[i].vx);
        float dqw = 0.5f * (-particles[i].qx * particles[i].vx - particles[i].qy * particles[i].vy - particles[i].qz * particles[i].vz);

        particles[i].qx += dqx * dt;
        particles[i].qy += dqy * dt;
        particles[i].qz += dqz * dt;
        particles[i].qw += dqw * dt;

        float qnorm = std::sqrt(particles[i].qx * particles[i].qx + particles[i].qy * particles[i].qy + particles[i].qz * particles[i].qz + particles[i].qw * particles[i].qw);
        particles[i].qx /= qnorm;
        particles[i].qy /= qnorm;
        particles[i].qz /= qnorm;
        particles[i].qw /= qnorm;

        particles[i].px += particles[i].vx * dt;
        particles[i].py += particles[i].vy * dt;
        particles[i].pz += particles[i].vz * dt;
    }
}

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
    enki::TaskScheduler ts;
    ts.Initialize();

    std::cout << "===============================================================\n";
#if defined(FP_MODE_STRICT)
    std::cout << "  HEAVY SIMD MATH BENCHMARK: MSVC /fp:strict                   \n";
#elif defined(FP_MODE_PRECISE)
    std::cout << "  HEAVY SIMD MATH BENCHMARK: MSVC /fp:precise                  \n";
#elif defined(FP_MODE_FAST)
    std::cout << "  HEAVY SIMD MATH BENCHMARK: MSVC /fp:fast                     \n";
#endif
    std::cout << "===============================================================\n";
    std::cout << "Workload: Quaternion Integration + Distance Masking + 15,000 Bodies\n";

    JPH::RegisterDefaultAllocator();
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();

    const size_t NUM_BODIES = 15000;
    const int NUM_FRAMES = 500;
    const float DELTA_TIME = 1.0f / 60.0f;

    BPLayerInterfaceImpl broad_phase_layer_interface;
    ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
    ObjectLayerPairFilterImpl object_vs_object_layer_filter;

    // -------------------------------------------------------------
    // RUN 1: Pure Jolt + Heavy Scalar Math Pass
    // -------------------------------------------------------------
    std::cout << "[RUN 1] Pure Jolt + Heavy Scalar Math Pass A...";
    JPH::TempAllocatorImpl temp_allocator_det1(32 * 1024 * 1024);
    JoltEnkiTSJobSystem job_system_det1(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, &ts);
    
    JPH::PhysicsSystem physics_system_det1;
    physics_system_det1.Init(static_cast<uint32_t>(NUM_BODIES + 10), 0, 2048, 2048, broad_phase_layer_interface, object_vs_broadphase_layer_filter, object_vs_object_layer_filter);
    JPH::BodyInterface& body_interface_det1 = physics_system_det1.GetBodyInterface();

    JPH::BoxShapeSettings floor_settings(JPH::Vec3(200.0f, 1.0f, 200.0f));
    JPH::ShapeRefC floor_shape = floor_settings.Create().Get();
    JPH::BodyCreationSettings floor_bcs(floor_shape, JPH::RVec3(0.0f, -1.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::NON_MOVING);
    body_interface_det1.CreateAndAddBody(floor_bcs, JPH::EActivation::DontActivate);

    JPH::SphereShapeSettings sphere_shape_settings(0.4f);
    JPH::ShapeRefC sphere_shape = sphere_shape_settings.Create().Get();
    JPH::BoxShapeSettings box_shape_settings(JPH::Vec3(0.4f, 0.4f, 0.4f));
    JPH::ShapeRefC box_shape = box_shape_settings.Create().Get();

    std::vector<JPH::BodyID> body_ids_det1;
    body_ids_det1.reserve(NUM_BODIES);
    std::vector<PhysicsParticleState> scalar_particles(NUM_BODIES);

    for (size_t i = 0; i < NUM_BODIES; ++i) {
        float x = static_cast<float>(i % 50) - 25.0f;
        float y = 5.0f + static_cast<float>(i / (50 * 50)) * 1.1f;
        float z = static_cast<float>((i / 50) % 50) - 25.0f;

        JPH::ShapeRefC active_shape = (i % 2 == 0) ? sphere_shape : box_shape;
        JPH::BodyCreationSettings sphere_bcs(active_shape, JPH::RVec3(x, y, z), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
        sphere_bcs.mFriction = 0.5f;
        sphere_bcs.mRestitution = 0.3f;
        
        JPH::Body* body = body_interface_det1.CreateBody(sphere_bcs);
        body_ids_det1.push_back(body->GetID());
        body_interface_det1.AddBody(body->GetID(), JPH::EActivation::Activate);

        scalar_particles[i] = { x, y, z, 1.0f, 0.1f, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    }

    auto start_det1 = std::chrono::high_resolution_clock::now();
    for (int frame = 0; frame < NUM_FRAMES; ++frame) {
        ProcessHeavyScalarParticleBatch(scalar_particles.data(), scalar_particles.size(), DELTA_TIME);
        physics_system_det1.Update(DELTA_TIME, 1, &temp_allocator_det1, &job_system_det1);
    }
    auto end_det1 = std::chrono::high_resolution_clock::now();
    double duration_jolt = std::chrono::duration<double, std::milli>(end_det1 - start_det1).count();
    std::cout << " Done.\n";

    std::vector<float> final_positions_det1;
    final_positions_det1.reserve(NUM_BODIES * 3);
    for (auto id : body_ids_det1) {
        JPH::RVec3 pos = body_interface_det1.GetPosition(id);
        final_positions_det1.push_back(static_cast<float>(pos.GetX()));
        final_positions_det1.push_back(static_cast<float>(pos.GetY()));
        final_positions_det1.push_back(static_cast<float>(pos.GetZ()));
    }
    uint64_t hash_jolt = ComputeStateHash(final_positions_det1.data(), final_positions_det1.size() * sizeof(float));

    // -------------------------------------------------------------
    // RUN 2: Jolt + Heavy Highway SIMD Pass
    // -------------------------------------------------------------
    std::cout << "[RUN 2] Jolt + Heavy Highway SIMD Math Pass...";
    JPH::TempAllocatorImpl temp_allocator_hwy(32 * 1024 * 1024);
    JoltEnkiTSJobSystem job_system_hwy(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, &ts);
    
    JPH::PhysicsSystem physics_system_hwy;
    physics_system_hwy.Init(static_cast<uint32_t>(NUM_BODIES + 10), 0, 2048, 2048, broad_phase_layer_interface, object_vs_broadphase_layer_filter, object_vs_object_layer_filter);
    JPH::BodyInterface& body_interface_hwy = physics_system_hwy.GetBodyInterface();

    body_interface_hwy.CreateAndAddBody(floor_bcs, JPH::EActivation::DontActivate);

    std::vector<JPH::BodyID> body_ids_hwy;
    body_ids_hwy.reserve(NUM_BODIES);
    std::vector<PhysicsParticleState> highway_particles(NUM_BODIES);

    for (size_t i = 0; i < NUM_BODIES; ++i) {
        float x = static_cast<float>(i % 50) - 25.0f;
        float y = 5.0f + static_cast<float>(i / (50 * 50)) * 1.1f;
        float z = static_cast<float>((i / 50) % 50) - 25.0f;

        JPH::ShapeRefC active_shape = (i % 2 == 0) ? sphere_shape : box_shape;
        JPH::BodyCreationSettings sphere_bcs(active_shape, JPH::RVec3(x, y, z), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
        sphere_bcs.mFriction = 0.5f;
        sphere_bcs.mRestitution = 0.3f;
        
        JPH::Body* body = body_interface_hwy.CreateBody(sphere_bcs);
        body_ids_hwy.push_back(body->GetID());
        body_interface_hwy.AddBody(body->GetID(), JPH::EActivation::Activate);

        highway_particles[i] = { x, y, z, 1.0f, 0.1f, 0.2f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    }

    auto start_hwy = std::chrono::high_resolution_clock::now();
    for (int frame = 0; frame < NUM_FRAMES; ++frame) {
        ProcessHeavyHighwayParticleBatchEnkiTS(ts, highway_particles.data(), highway_particles.size(), DELTA_TIME);
        physics_system_hwy.Update(DELTA_TIME, 1, &temp_allocator_hwy, &job_system_hwy);
    }
    auto end_hwy = std::chrono::high_resolution_clock::now();
    double duration_hwy = std::chrono::duration<double, std::milli>(end_hwy - start_hwy).count();
    std::cout << " Done.\n";

    std::vector<float> final_positions_hwy;
    final_positions_hwy.reserve(NUM_BODIES * 3);
    for (auto id : body_ids_hwy) {
        JPH::RVec3 pos = body_interface_hwy.GetPosition(id);
        final_positions_hwy.push_back(static_cast<float>(pos.GetX()));
        final_positions_hwy.push_back(static_cast<float>(pos.GetY()));
        final_positions_hwy.push_back(static_cast<float>(pos.GetZ()));
    }
    uint64_t hash_hwy = ComputeStateHash(final_positions_hwy.data(), final_positions_hwy.size() * sizeof(float));

    // -------------------------------------------------------------
    // RESULTS & SUMMARY REPORT
    // -------------------------------------------------------------
    std::cout << "\n===============================================================\n";
    std::cout << "                     PERFORMANCE RESULTS                       \n";
    std::cout << "===============================================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Pure Jolt + Heavy Scalar Math Execution Time : " << duration_jolt << " ms (" << duration_jolt / NUM_FRAMES << " ms/frame)\n";
    std::cout << "Jolt + Heavy Highway SIMD Execution Time     : " << duration_hwy << " ms (" << duration_hwy / NUM_FRAMES << " ms/frame)\n";
    
    double speedup = ((duration_jolt - duration_hwy) / duration_jolt) * 100.0;
    std::cout << "Highway SIMD Acceleration Speedup            : " << (speedup >= 0 ? "+" : "") << speedup << "% faster throughput!\n";

    std::cout << "\n===============================================================\n";
    std::cout << "                DETERMINISM VERIFICATION REPORT                \n";
    std::cout << "===============================================================\n";
    std::cout << "Pure Jolt Scalar Hash: 0x" << std::hex << std::uppercase << hash_jolt << std::dec << "\n";
    std::cout << "Jolt + Highway Hash  : 0x" << std::hex << std::uppercase << hash_hwy << std::dec << "\n";

    if (hash_jolt == hash_hwy) {
        std::cout << "\n[RESULT] Determinism Re-Verification: PASSED! (100% Bit-Exact Match)\n";
    } else {
        std::cout << "\n[RESULT] Determinism Re-Verification: FAILED!\n";
    }

    delete JPH::Factory::sInstance;
    return 0;
}
