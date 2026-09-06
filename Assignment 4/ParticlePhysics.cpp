#ifndef NOMINMAX
#define NOMINMAX
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ParticlePhysics.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "IParticlePhysicsPlugin.hpp"
#include <vector>
#include <algorithm>
#include <execution>
#include <iostream>

// Backend concurrency library headers
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <TaskScheduler.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <marl/scheduler.h>
#include <marl/waitgroup.h>
#include <marl/defer.h>
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

// Cross-platform DLL export macro
#if defined(_WIN32)
    #define EXPORT_API __declspec(dllexport)
#else
    #define EXPORT_API __attribute__((visibility("default")))
#endif

HWY_BEFORE_NAMESPACE();
namespace ParticlePhysicsPlugin {
namespace HWY_NAMESPACE {
    namespace hn = hwy::HWY_NAMESPACE;

    void VectorUpdate(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt) {
        const hn::ScalableTag<float> d;
        const size_t lanes = hn::Lanes(d);

        // Softening factor to prevent divide-by-zero or infinite gravity
        const auto v_eps_sq = hn::Set(d, 0.01f);
        const auto v_dt = hn::Set(d, dt);
        const auto v_g = hn::Set(d, 0.01f); // scaled gravity constant
        const auto v_g_dt = hn::Mul(v_g, v_dt);

        size_t i = 0;
        for (; i + lanes <= count; i += lanes) {
            hn::Vec<decltype(d)> px, py, pz, pw;
            hn::Vec<decltype(d)> vx, vy, vz, vw;

            // Load position and velocity blocks
            hn::LoadInterleaved4(d, reinterpret_cast<const float*>(&particles[i]), px, py, pz, pw);
            hn::LoadInterleaved4(d, reinterpret_cast<const float*>(&velocities[i]), vx, vy, vz, vw);

            // Accumulate gravitational forces
            auto ax = hn::Zero(d);
            auto ay = hn::Zero(d);
            auto az = hn::Zero(d);

            for (size_t j = 0; j < attractorCount; ++j) {
                auto attr_x = hn::Set(d, attractors[j].x);
                auto attr_y = hn::Set(d, attractors[j].y);
                auto attr_z = hn::Set(d, attractors[j].z);
                auto attr_mass = hn::Set(d, attractors[j].mass);

                // Delta vector
                auto dx = hn::Sub(attr_x, px);
                auto dy = hn::Sub(attr_y, py);
                auto dz = hn::Sub(attr_z, pz);

                // Distance squared + softening
                auto dist_sq = hn::MulAdd(dx, dx, hn::MulAdd(dy, dy, hn::MulAdd(dz, dz, v_eps_sq)));

                // inv_dist = rsqrt(dist_sq)
                auto inv_dist = hn::ApproximateReciprocalSqrt(dist_sq);

                // force = mass * inv_dist^3
                auto inv_dist3 = hn::Mul(inv_dist, hn::Mul(inv_dist, inv_dist));
                auto force = hn::Mul(attr_mass, inv_dist3);

                // Accumulate acceleration
                ax = hn::MulAdd(dx, force, ax);
                ay = hn::MulAdd(dy, force, ay);
                az = hn::MulAdd(dz, force, az);
            }

            // Integrate acceleration to velocity
            vx = hn::MulAdd(ax, v_g_dt, vx);
            vy = hn::MulAdd(ay, v_g_dt, vy);
            vz = hn::MulAdd(az, v_g_dt, vz);

            // Integrate velocity to position
            px = hn::MulAdd(vx, v_dt, px);
            py = hn::MulAdd(vy, v_dt, py);
            pz = hn::MulAdd(vz, v_dt, pz);

            // Store back to memory
            hn::StoreInterleaved4(px, py, pz, pw, d, reinterpret_cast<float*>(&particles[i]));
            hn::StoreInterleaved4(vx, vy, vz, vw, d, reinterpret_cast<float*>(&velocities[i]));
        }

        // Tail-end scalar fallback
        for (; i < count; ++i) {
            float px = particles[i].x;
            float py = particles[i].y;
            float pz = particles[i].z;

            float vx = velocities[i].vx;
            float vy = velocities[i].vy;
            float vz = velocities[i].vz;

            float ax = 0.0f;
            float ay = 0.0f;
            float az = 0.0f;

            for (size_t j = 0; j < attractorCount; ++j) {
                float dx = attractors[j].x - px;
                float dy = attractors[j].y - py;
                float dz = attractors[j].z - pz;

                float dist_sq = dx * dx + dy * dy + dz * dz + 0.01f;
                float inv_dist = 1.0f / std::sqrt(dist_sq);
                float force = attractors[j].mass * inv_dist * inv_dist * inv_dist;

                ax += dx * force;
                ay += dy * force;
                az += dz * force;
            }

            vx += ax * 0.01f * dt;
            vy += ay * 0.01f * dt;
            vz += az * 0.01f * dt;

            px += vx * dt;
            py += vy * dt;
            pz += vz * dt;

            particles[i].x = px;
            particles[i].y = py;
            particles[i].z = pz;

            velocities[i].vx = vx;
            velocities[i].vy = vy;
            velocities[i].vz = vz;
        }
    }
} // namespace HWY_NAMESPACE
} // namespace ParticlePhysicsPlugin
HWY_AFTER_NAMESPACE();

#include <hwy/foreach_target.h>

#if HWY_ONCE
namespace ParticlePhysicsPlugin {
    // Generate dyn-dispatch pointer hook
    HWY_EXPORT(VectorUpdate);

    // Custom enkiTS task set definition
    struct EnkiPhysicsTask : public enki::TaskSet {
        MeshInstanceData* particles;
        MeshInstanceVelocity* velocities;
        const AttractorData* attractors;
        size_t attractorCount;
        float dt;
        EnkiPhysicsTask(MeshInstanceData* p, MeshInstanceVelocity* v, const AttractorData* a, size_t ac, float t) 
            : particles(p), velocities(v), attractors(a), attractorCount(ac), dt(t) {}
        
        void ExecuteRange(enki::TaskSetPartition range, uint32_t threadnum) override {
            HWY_DYNAMIC_DISPATCH(VectorUpdate)(
                &particles[range.start], 
                &velocities[range.start], 
                range.end - range.start, 
                attractors, 
                attractorCount, 
                dt
            );
        }
    };

    class ParticlePhysicsPluginImpl : public IParticlePhysicsPlugin {
    public:
        std::unique_ptr<marl::Scheduler> marlScheduler;

        bool Initialize(VkInstance instance, VkDevice device) override {
            std::cout << "[ParticlePhysics DLL] Compute-bound SIMD Attractor Engine Initialized." << std::endl;
            marlScheduler = std::make_unique<marl::Scheduler>(marl::Scheduler::Config::allCores());
            marlScheduler->bind();
            return true;
        }

        void Shutdown() override {
            if (marlScheduler) {
                marlScheduler->unbind();
                marlScheduler.reset();
            }
            std::cout << "[ParticlePhysics DLL] Dynamic SIMD Engine Shutdown." << std::endl;
        }

        void Update() override {}

        // Legacy/Fallback implementation
        void UpdateParticles(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt) override {
            UpdateParticlesTBB(particles, velocities, count, attractors, attractorCount, dt);
        }

        void UpdateParticlesSingleThreaded(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt) override {
            HWY_DYNAMIC_DISPATCH(VectorUpdate)(particles, velocities, count, attractors, attractorCount, dt);
        }

        // 1. oneTBB integration
        void UpdateParticlesTBB(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt) override {
            const size_t grainSize = 4096; // Smaller grain size since each particle task has high compute density
            tbb::parallel_for(tbb::blocked_range<size_t>(0, count, grainSize), 
                [=](const tbb::blocked_range<size_t>& range) {
                    HWY_DYNAMIC_DISPATCH(VectorUpdate)(
                        &particles[range.begin()], 
                        &velocities[range.begin()], 
                        range.end() - range.begin(), 
                        attractors,
                        attractorCount,
                        dt
                    );
                }, 
                tbb::simple_partitioner{}
            );
        }

        // 2. enkiTS integration
        void UpdateParticlesEnki(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt, void* scheduler) override {
            auto* enkiScheduler = static_cast<enki::TaskScheduler*>(scheduler);
            EnkiPhysicsTask task(particles, velocities, attractors, attractorCount, dt);
            task.m_SetSize = static_cast<uint32_t>(count);
            task.m_MinRange = 4096; // Grain size control
            enkiScheduler->AddTaskSetToPipe(&task);
            enkiScheduler->WaitforTask(&task);
        }

        // 3. Taskflow integration
        void UpdateParticlesTaskflow(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt, void* executor) override {
            auto* tfExecutor = static_cast<tf::Executor*>(executor);
            tf::Taskflow taskflow;
            const size_t chunkSize = 4096;
            taskflow.for_each_index(size_t(0), count, chunkSize, [=](size_t index) {
                size_t end = std::min(index + chunkSize, count);
                HWY_DYNAMIC_DISPATCH(VectorUpdate)(
                    &particles[index], 
                    &velocities[index], 
                    end - index, 
                    attractors,
                    attractorCount,
                    dt
                );
            });
            tfExecutor->run(taskflow).wait();
        }

        // 4. Marl integration
        void UpdateParticlesMarl(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt, void* /*unused*/) override {
            const size_t chunkSize = 4096;
            const size_t totalChunks = (count + chunkSize - 1) / chunkSize;
            marl::WaitGroup wg(static_cast<uint32_t>(totalChunks));

            for (size_t chunkIdx = 0; chunkIdx < totalChunks; ++chunkIdx) {
                marl::schedule([=]() mutable {
                    defer(wg.done());
                    size_t start = chunkIdx * chunkSize;
                    size_t end = std::min(start + chunkSize, count);
                    HWY_DYNAMIC_DISPATCH(VectorUpdate)(
                        &particles[start], 
                        &velocities[start], 
                        end - start, 
                        attractors,
                        attractorCount,
                        dt
                    );
                });
            }
            wg.wait();
        }

        // 5. stdexec integration
        void UpdateParticlesStdexec(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt, void* pool) override {
            auto* threadPool = static_cast<exec::static_thread_pool*>(pool);
            auto scheduler = threadPool->get_scheduler();
            const size_t chunkSize = 4096;
            const size_t totalChunks = (count + chunkSize - 1) / chunkSize;

            auto work = stdexec::just()
                | stdexec::transfer(scheduler)
                | stdexec::bulk(totalChunks, [=](size_t chunkIdx) {
                    size_t start = chunkIdx * chunkSize;
                    size_t end = std::min(start + chunkSize, count);
                    HWY_DYNAMIC_DISPATCH(VectorUpdate)(
                        &particles[start], 
                        &velocities[start], 
                        end - start, 
                        attractors,
                        attractorCount,
                        dt
                    );
                });
            stdexec::sync_wait(work);
        }
    };
} // namespace ParticlePhysicsPlugin

// Export factory function
extern "C" EXPORT_API IFeaturePlugin* CreateDynamicPlugin() {
    return new ParticlePhysicsPlugin::ParticlePhysicsPluginImpl();
}
#endif
