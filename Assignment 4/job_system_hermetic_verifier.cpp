#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cmath>

// enkiTS
#include "TaskScheduler.h"

// Taskflow
#include <taskflow/taskflow.hpp>

// Google Highway SIMD
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

struct alignas(16) Particle {
    float x, y, z, w;
    float vx, vy, vz, vw;
};

// High-precision FNV-1a 64-bit Hash
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

// SIMD Kernel
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

struct StatResult {
    double minMs;
    double maxMs;
    double meanMs;
    double stdDevMs;
    uint64_t finalHash;
};

StatResult RunBenchmarkTrial(const std::vector<Particle>& initialParticles, size_t particleCount, size_t chunkSize, int frames, float dt, int trials, std::function<uint64_t(std::vector<Particle>&)> runFrameLoop) {
    std::vector<double> trialTimes;
    trialTimes.reserve(trials);
    uint64_t lastHash = 0;

    for (int t = 0; t < trials; ++t) {
        std::vector<Particle> particles = initialParticles; // Fresh isolated copy per trial

        // Warmup pass (10 frames)
        for (int w = 0; w < 10; ++w) {
            runFrameLoop(particles);
        }

        particles = initialParticles; // Reset state post warmup

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int f = 0; f < frames; ++f) {
            lastHash = runFrameLoop(particles);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        trialTimes.push_back(elapsed);
    }

    double sum = std::accumulate(trialTimes.begin(), trialTimes.end(), 0.0);
    double mean = sum / trials;

    double sq_sum = 0.0;
    for (double timeMs : trialTimes) {
        sq_sum += (timeMs - mean) * (timeMs - mean);
    }
    double stdDev = std::sqrt(sq_sum / trials);

    double minVal = *std::min_element(trialTimes.begin(), trialTimes.end());
    double maxVal = *std::max_element(trialTimes.begin(), trialTimes.end());

    return { minVal, maxVal, mean, stdDev, lastHash };
}

int main() {
    constexpr size_t PARTICLE_COUNT = 250'000;
    constexpr int FRAMES = 50;
    constexpr float DT = 0.01667f;
    constexpr int TRIALS = 3;

    const std::vector<size_t> chunkSizes = { 1024, 4096, 16384 };


    std::cout << "=========================================================================================================\n";
    std::cout << "  HERMETIC JOB SYSTEM EVALUATION: enkiTS vs TASKFLOW RIGOROUS CRITIQUE & BENCHMARK\n";
    std::cout << "  Particles: " << PARTICLE_COUNT << " | Frames per Run: " << FRAMES << " | Trials: " << TRIALS << "\n";
    std::cout << "=========================================================================================================\n\n";

    std::vector<Particle> initialParticles;
    InitParticles(initialParticles, PARTICLE_COUNT);

    std::cout << std::left 
              << std::setw(12) << "Chunk Size" 
              << std::setw(22) << "Engine / Mode" 
              << std::setw(12) << "Mean (ms)" 
              << std::setw(12) << "Min (ms)" 
              << std::setw(12) << "Max (ms)" 
              << std::setw(12) << "StdDev" 
              << std::setw(20) << "State Hash" 
              << "\n";
    std::cout << "---------------------------------------------------------------------------------------------------------\n";

    for (size_t chunkSize : chunkSizes) {
        size_t totalChunks = (PARTICLE_COUNT + chunkSize - 1) / chunkSize;

        // ---------------------------------------------------------------------------------
        // 1. enkiTS Multi-threaded (Work-Stealing)
        // ---------------------------------------------------------------------------------
        {
            enki::TaskScheduler ts;
            ts.Initialize();

            auto runner = [&](std::vector<Particle>& particles) -> uint64_t {
                enki::TaskSet task(static_cast<uint32_t>(totalChunks), [&](enki::TaskSetPartition range, uint32_t threadnum) {
                    for (uint32_t c = range.start; c < range.end; ++c) {
                        size_t startIdx = c * chunkSize;
                        size_t endIdx = std::min(startIdx + chunkSize, PARTICLE_COUNT);
                        ProcessParticleChunk(particles.data(), startIdx, endIdx, DT);
                    }
                });
                ts.AddTaskSetToPipe(&task);
                ts.WaitforTask(&task);
                return ComputeStateHash(particles);
            };

            StatResult res = RunBenchmarkTrial(initialParticles, PARTICLE_COUNT, chunkSize, FRAMES, DT, TRIALS, runner);
            std::cout << std::left 
                      << std::setw(12) << chunkSize 
                      << std::setw(22) << "enkiTS (Work-Steal)" 
                      << std::setw(12) << std::fixed << std::setprecision(2) << res.meanMs 
                      << std::setw(12) << res.minMs 
                      << std::setw(12) << res.maxMs 
                      << std::setw(12) << res.stdDevMs 
                      << "0x" << std::hex << std::uppercase << res.finalHash << std::dec << "\n";
        }

        // ---------------------------------------------------------------------------------
        // 2. enkiTS Single-Threaded (Deterministic Mode)
        // ---------------------------------------------------------------------------------
        {
            enki::TaskSchedulerConfig config;
            config.numTaskThreadsToCreate = 0;
            enki::TaskScheduler ts;
            ts.Initialize(config);

            auto runner = [&](std::vector<Particle>& particles) -> uint64_t {
                enki::TaskSet task(static_cast<uint32_t>(totalChunks), [&](enki::TaskSetPartition range, uint32_t threadnum) {
                    for (uint32_t c = range.start; c < range.end; ++c) {
                        size_t startIdx = c * chunkSize;
                        size_t endIdx = std::min(startIdx + chunkSize, PARTICLE_COUNT);
                        ProcessParticleChunk(particles.data(), startIdx, endIdx, DT);
                    }
                });
                ts.AddTaskSetToPipe(&task);
                ts.WaitforTask(&task);
                return ComputeStateHash(particles);
            };

            StatResult res = RunBenchmarkTrial(initialParticles, PARTICLE_COUNT, chunkSize, FRAMES, DT, TRIALS, runner);
            std::cout << std::left 
                      << std::setw(12) << chunkSize 
                      << std::setw(22) << "enkiTS (Deterministic)" 
                      << std::setw(12) << std::fixed << std::setprecision(2) << res.meanMs 
                      << std::setw(12) << res.minMs 
                      << std::setw(12) << res.maxMs 
                      << std::setw(12) << res.stdDevMs 
                      << "0x" << std::hex << std::uppercase << res.finalHash << std::dec << "\n";
        }

        // ---------------------------------------------------------------------------------
        // 3. Taskflow Multi-threaded (Zero-Allocation Pre-constructed Task Graph)
        // ---------------------------------------------------------------------------------
        {
            tf::Executor executor;

            auto runner = [&](std::vector<Particle>& particles) -> uint64_t {
                tf::Taskflow taskflow;
                for (size_t c = 0; c < totalChunks; ++c) {
                    size_t startIdx = c * chunkSize;
                    size_t endIdx = std::min(startIdx + chunkSize, PARTICLE_COUNT);
                    taskflow.emplace([&particles, startIdx, endIdx, DT]() {
                        ProcessParticleChunk(particles.data(), startIdx, endIdx, DT);
                    });
                }
                executor.run(taskflow).wait();
                return ComputeStateHash(particles);
            };

            StatResult res = RunBenchmarkTrial(initialParticles, PARTICLE_COUNT, chunkSize, FRAMES, DT, TRIALS, runner);
            std::cout << std::left 
                      << std::setw(12) << chunkSize 
                      << std::setw(22) << "Taskflow (Work-Steal)" 
                      << std::setw(12) << std::fixed << std::setprecision(2) << res.meanMs 
                      << std::setw(12) << res.minMs 
                      << std::setw(12) << res.maxMs 
                      << std::setw(12) << res.stdDevMs 
                      << "0x" << std::hex << std::uppercase << res.finalHash << std::dec << "\n";
        }

        // ---------------------------------------------------------------------------------
        // 4. Taskflow Single-Threaded (Deterministic Mode)
        // ---------------------------------------------------------------------------------
        {
            tf::Executor executor(1);

            auto runner = [&](std::vector<Particle>& particles) -> uint64_t {
                tf::Taskflow taskflow;
                for (size_t c = 0; c < totalChunks; ++c) {
                    size_t startIdx = c * chunkSize;
                    size_t endIdx = std::min(startIdx + chunkSize, PARTICLE_COUNT);
                    taskflow.emplace([&particles, startIdx, endIdx, DT]() {
                        ProcessParticleChunk(particles.data(), startIdx, endIdx, DT);
                    });
                }
                executor.run(taskflow).wait();
                return ComputeStateHash(particles);
            };

            StatResult res = RunBenchmarkTrial(initialParticles, PARTICLE_COUNT, chunkSize, FRAMES, DT, TRIALS, runner);
            std::cout << std::left 
                      << std::setw(12) << chunkSize 
                      << std::setw(22) << "Taskflow (Deterministic)" 
                      << std::setw(12) << std::fixed << std::setprecision(2) << res.meanMs 
                      << std::setw(12) << res.minMs 
                      << std::setw(12) << res.maxMs 
                      << std::setw(12) << res.stdDevMs 
                      << "0x" << std::hex << std::uppercase << res.finalHash << std::dec << "\n";
        }

        std::cout << "---------------------------------------------------------------------------------------------------------\n";
    }

    return 0;
}
