#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cstdint>

// enkiTS
#include "TaskScheduler.h"

// Taskflow
#include <taskflow/taskflow.hpp>

// Highway SIMD
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

struct Particle {
    float x, y, z, w;
    float vx, vy, vz, vw;
};

// 64-bit FNV-1a hash function for state determinism check
uint64_t ComputeStateHash(const std::vector<Particle>& particles) {
    uint64_t hash = 14695981039346656037ULL;
    const uint64_t prime = 1099511628211ULL;
    const uint8_t* byteData = reinterpret_cast<const uint8_t*>(particles.data());
    size_t sizeBytes = particles.size() * sizeof(Particle);

    for (size_t i = 0; i < sizeBytes; ++i) {
        hash ^= byteData[i];
        hash *= prime;
    }
    return hash;
}

// Particle update SIMD kernel
void ProcessParticleChunk(Particle* data, size_t start, size_t end, float dt) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const auto v_dt = hn::Set(d, dt);
    const auto v_gravity = hn::Set(d, -9.81f);

    size_t i = start;
    for (; i + lanes <= end; i += lanes) {
        float px[16], py[16], pz[16];
        float vx[16], vy[16], vz[16];

        for (size_t l = 0; l < lanes; ++l) {
            px[l] = data[i + l].x;
            py[l] = data[i + l].y;
            pz[l] = data[i + l].z;
            vx[l] = data[i + l].vx;
            vy[l] = data[i + l].vy;
            vz[l] = data[i + l].vz;
        }

        auto v_px = hn::LoadU(d, px);
        auto v_py = hn::LoadU(d, py);
        auto v_pz = hn::LoadU(d, pz);
        auto v_vx = hn::LoadU(d, vx);
        auto v_vy = hn::LoadU(d, vy);
        auto v_vz = hn::LoadU(d, vz);

        v_vy = hn::MulAdd(v_gravity, v_dt, v_vy);
        v_px = hn::MulAdd(v_vx, v_dt, v_px);
        v_py = hn::MulAdd(v_vy, v_dt, v_py);
        v_pz = hn::MulAdd(v_vz, v_dt, v_pz);

        hn::StoreU(v_vx, d, vx);
        hn::StoreU(v_vy, d, vy);
        hn::StoreU(v_vz, d, vz);
        hn::StoreU(v_px, d, px);
        hn::StoreU(v_py, d, py);
        hn::StoreU(v_pz, d, pz);

        for (size_t l = 0; l < lanes; ++l) {
            data[i + l].x = px[l];
            data[i + l].y = py[l];
            data[i + l].z = pz[l];
            data[i + l].vx = vx[l];
            data[i + l].vy = vy[l];
            data[i + l].vz = vz[l];
        }
    }

    for (; i < end; ++i) {
        data[i].vy += -9.81f * dt;
        data[i].x += data[i].vx * dt;
        data[i].y += data[i].vy * dt;
        data[i].z += data[i].vz * dt;
    }
}

void InitParticles(std::vector<Particle>& particles, size_t count) {
    particles.resize(count);
    for (size_t i = 0; i < count; ++i) {
        particles[i] = {
            static_cast<float>(i % 100),
            static_cast<float>((i / 100) % 100),
            static_cast<float>(i / 10000),
            1.0f,
            1.0f, 2.0f, 0.5f, 0.0f
        };
    }
}

int main() {
    constexpr size_t PARTICLE_COUNT = 500'000;
    constexpr size_t CHUNK_SIZE = 4096;
    constexpr int FRAMES = 100;
    constexpr float DT = 0.01667f;

    std::cout << "=================================================================================\n";
    std::cout << "  JOB SYSTEM EVALUATION BENCHMARK: WORK-STEALING VS DETERMINISTIC EXECUTION\n";
    std::cout << "  Particles: " << PARTICLE_COUNT << " | Frames: " << FRAMES << "\n";
    std::cout << "=================================================================================\n\n";

    std::vector<Particle> initialParticles;
    InitParticles(initialParticles, PARTICLE_COUNT);

    // ---------------------------------------------------------------------------------
    // 1. enkiTS Multi-threaded Work-Stealing Mode
    // ---------------------------------------------------------------------------------
    {
        std::vector<Particle> particles = initialParticles;
        enki::TaskScheduler ts;
        ts.Initialize();

        auto start = std::chrono::high_resolution_clock::now();

        for (int f = 0; f < FRAMES; ++f) {
            size_t totalChunks = (PARTICLE_COUNT + CHUNK_SIZE - 1) / CHUNK_SIZE;
            enki::TaskSet task(static_cast<uint32_t>(totalChunks), [&](enki::TaskSetPartition range, uint32_t threadnum) {
                for (uint32_t c = range.start; c < range.end; ++c) {
                    size_t startIdx = c * CHUNK_SIZE;
                    size_t endIdx = std::min(startIdx + CHUNK_SIZE, PARTICLE_COUNT);
                    ProcessParticleChunk(particles.data(), startIdx, endIdx, DT);
                }
            });
            ts.AddTaskSetToPipe(&task);
            ts.WaitforTask(&task);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
        uint64_t hash = ComputeStateHash(particles);

        std::cout << std::left << std::setw(45) << "[enkiTS] Multi-threaded (Work-Stealing):"
                  << std::setw(12) << elapsedMs << " ms | Hash: 0x" << std::hex << std::uppercase << hash << std::dec << "\n";
    }

    // ---------------------------------------------------------------------------------
    // 2. enkiTS Deterministic Mode (Single-Threaded Worker Execution)
    // ---------------------------------------------------------------------------------
    uint64_t enkiTS_det_hash_1 = 0;
    uint64_t enkiTS_det_hash_2 = 0;
    {
        for (int run = 0; run < 2; ++run) {
            std::vector<Particle> particles = initialParticles;
            enki::TaskSchedulerConfig config;
            config.numTaskThreadsToCreate = 0; // Force 0 extra worker threads -> execution on main thread strictly
            enki::TaskScheduler ts;
            ts.Initialize(config);

            auto start = std::chrono::high_resolution_clock::now();

            for (int f = 0; f < FRAMES; ++f) {
                size_t totalChunks = (PARTICLE_COUNT + CHUNK_SIZE - 1) / CHUNK_SIZE;
                enki::TaskSet task(static_cast<uint32_t>(totalChunks), [&](enki::TaskSetPartition range, uint32_t threadnum) {
                    for (uint32_t c = range.start; c < range.end; ++c) {
                        size_t startIdx = c * CHUNK_SIZE;
                        size_t endIdx = std::min(startIdx + CHUNK_SIZE, PARTICLE_COUNT);
                        ProcessParticleChunk(particles.data(), startIdx, endIdx, DT);
                    }
                });
                ts.AddTaskSetToPipe(&task);
                ts.WaitforTask(&task);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
            uint64_t hash = ComputeStateHash(particles);
            if (run == 0) enkiTS_det_hash_1 = hash;
            else enkiTS_det_hash_2 = hash;

            std::cout << std::left << std::setw(45) << (run == 0 ? "[enkiTS] Deterministic Single-Threaded Pass 1:" : "[enkiTS] Deterministic Single-Threaded Pass 2:")
                      << std::setw(12) << elapsedMs << " ms | Hash: 0x" << std::hex << std::uppercase << hash << std::dec << "\n";
        }
    }

    // ---------------------------------------------------------------------------------
    // 3. Taskflow Multi-threaded Executor (Work-Stealing Topology)
    // ---------------------------------------------------------------------------------
    {
        std::vector<Particle> particles = initialParticles;
        tf::Executor executor;
        tf::Taskflow taskflow;

        size_t totalChunks = (PARTICLE_COUNT + CHUNK_SIZE - 1) / CHUNK_SIZE;
        for (size_t c = 0; c < totalChunks; ++c) {
            size_t startIdx = c * CHUNK_SIZE;
            size_t endIdx = std::min(startIdx + CHUNK_SIZE, PARTICLE_COUNT);
            taskflow.emplace([&particles, startIdx, endIdx]() {
                ProcessParticleChunk(particles.data(), startIdx, endIdx, DT);
            });
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (int f = 0; f < FRAMES; ++f) {
            executor.run(taskflow).wait();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
        uint64_t hash = ComputeStateHash(particles);

        std::cout << std::left << std::setw(45) << "[Taskflow] Multi-threaded (Work-Stealing):"
                  << std::setw(12) << elapsedMs << " ms | Hash: 0x" << std::hex << std::uppercase << hash << std::dec << "\n";
    }

    // ---------------------------------------------------------------------------------
    // 4. Taskflow Deterministic Mode (Single-Threaded Topologically Sorted Executor)
    // ---------------------------------------------------------------------------------
    uint64_t tf_det_hash_1 = 0;
    uint64_t tf_det_hash_2 = 0;
    {
        for (int run = 0; run < 2; ++run) {
            std::vector<Particle> particles = initialParticles;
            tf::Executor executor(1); // Force exactly 1 thread executor for deterministic sequential dispatch
            tf::Taskflow taskflow;

            size_t totalChunks = (PARTICLE_COUNT + CHUNK_SIZE - 1) / CHUNK_SIZE;
            for (size_t c = 0; c < totalChunks; ++c) {
                size_t startIdx = c * CHUNK_SIZE;
                size_t endIdx = std::min(startIdx + CHUNK_SIZE, PARTICLE_COUNT);
                taskflow.emplace([&particles, startIdx, endIdx]() {
                    ProcessParticleChunk(particles.data(), startIdx, endIdx, DT);
                });
            }

            auto start = std::chrono::high_resolution_clock::now();

            for (int f = 0; f < FRAMES; ++f) {
                executor.run(taskflow).wait();
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
            uint64_t hash = ComputeStateHash(particles);
            if (run == 0) tf_det_hash_1 = hash;
            else tf_det_hash_2 = hash;

            std::cout << std::left << std::setw(45) << (run == 0 ? "[Taskflow] Deterministic 1-Thread Pass 1:" : "[Taskflow] Deterministic 1-Thread Pass 2:")
                      << std::setw(12) << elapsedMs << " ms | Hash: 0x" << std::hex << std::uppercase << hash << std::dec << "\n";
        }
    }


    std::cout << "\n=================================================================================\n";
    std::cout << "  DETERMINISM VERIFICATION SUMMARY\n";
    std::cout << "=================================================================================\n";
    std::cout << "enkiTS Pass 1 Hash == Pass 2 Hash    : " << (enkiTS_det_hash_1 == enkiTS_det_hash_2 ? "PASSED (100% Deterministic)" : "FAILED") << "\n";
    std::cout << "Taskflow Pass 1 Hash == Pass 2 Hash  : " << (tf_det_hash_1 == tf_det_hash_2 ? "PASSED (100% Deterministic)" : "FAILED") << "\n";
    std::cout << "enkiTS Hash == Taskflow Hash        : " << (enkiTS_det_hash_1 == tf_det_hash_1 ? "MATCH (Cross-Engine Identical Outputs)" : "MISMATCH (Math order/chunking variation)") << "\n";
    std::cout << "=================================================================================\n";

    return 0;
}
