#pragma once

struct alignas(16) MeshInstanceData {
    float x, y, z, w;
};

struct alignas(16) MeshInstanceVelocity {
    float vx, vy, vz, vw;
};

struct alignas(16) AttractorData {
    float x, y, z, mass;
};
