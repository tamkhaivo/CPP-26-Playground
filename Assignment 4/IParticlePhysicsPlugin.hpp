#pragma once
#include "IFeaturePlugin.hpp"
#include "MeshInstanceData.hpp"
#include <cstddef>

struct IParticlePhysicsPlugin : public IFeaturePlugin {
    // Legacy base interfaces
    virtual void UpdateParticles(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt) = 0;
    virtual void UpdateParticlesSingleThreaded(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt) = 0;

    // Multi-core backend interfaces
    virtual void UpdateParticlesTBB(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt) = 0;
    virtual void UpdateParticlesEnki(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt, void* scheduler) = 0;
    virtual void UpdateParticlesTaskflow(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt, void* executor) = 0;
    virtual void UpdateParticlesMarl(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt, void* scheduler) = 0;
    virtual void UpdateParticlesStdexec(MeshInstanceData* particles, MeshInstanceVelocity* velocities, size_t count, const AttractorData* attractors, size_t attractorCount, float dt, void* pool) = 0;
};
