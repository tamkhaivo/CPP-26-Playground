#include <iostream>
#include <emmintrin.h>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <barrier>
#include <cassert>

namespace Benchmark {

// 64-bit FNV-1a Hash for Bit-Exact Verification
inline uint64_t FNV1a_64(const void* data, size_t size, uint64_t seed = 14695981039346656037ULL) {
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    uint64_t hash = seed;
    for (size_t i = 0; i < size; ++i) {
        hash ^= ptr[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Particle state conforming to std430 alignment rules
struct alignas(16) Particle {
    float posX, posY, posZ, pad0;
    float velX, velY, velZ, pad1;
    float mass, lifetime, flags, pad2;
};

// Simulation event (e.g., collision impact / particle spawn)
struct ImpactEvent {
    uint32_t particleId;
    float impactSpeed;
    float posX, posY, posZ;
};

inline void InitParticles(std::vector<Particle>& particles, size_t count) {
    particles.resize(count);
    std::memset(particles.data(), 0, count * sizeof(Particle));
    for (size_t i = 0; i < count; ++i) {
        particles[i].posX = static_cast<float>((i * 17) % 1000) - 500.0f;
        particles[i].posY = static_cast<float>((i * 31) % 500) + 10.0f;
        particles[i].posZ = static_cast<float>((i * 23) % 1000) - 500.0f;
        particles[i].velX = static_cast<float>((i * 7) % 50) - 25.0f;
        particles[i].velY = static_cast<float>((i * 13) % 60) - 30.0f;
        particles[i].velZ = static_cast<float>((i * 11) % 50) - 25.0f;
        particles[i].mass = 1.0f + static_cast<float>(i % 5);
        particles[i].lifetime = 100.0f;
        particles[i].flags = 1.0f;
    }
}

// Single particle integration step
inline bool UpdateParticle(Particle& p, float dt) {
    // Gravity + damping
    p.velY -= 9.81f * dt;
    p.velX *= 0.999f;
    p.velY *= 0.999f;
    p.velZ *= 0.999f;

    p.posX += p.velX * dt;
    p.posY += p.velY * dt;
    p.posZ += p.velZ * dt;

    bool impacted = false;
    // Ground bounce
    if (p.posY < 0.0f) {
        p.posY = -p.posY * 0.4f;
        p.velY = -p.velY * 0.5f;
        impacted = true;
    }
    // Lateral walls
    if (std::abs(p.posX) > 500.0f) {
        p.velX = -p.velX * 0.7f;
        impacted = true;
    }
    if (std::abs(p.posZ) > 500.0f) {
        p.velZ = -p.velZ * 0.7f;
        impacted = true;
    }
    return impacted;
}

// Result structure for scheduler run
struct SchedulerResult {
    double totalTimeMs;
    double dispatchLatencyUs;
    double threadJitterStdDevMs;
    double minThreadTimeMs;
    double maxThreadTimeMs;
    double throughputMeps; // Million entities per sec
    uint64_t stateHash;
    uint64_t eventBufferHash;
    size_t totalEvents;
};

// ============================================================================
// 1. DYNAMIC ATOMIC WORK-STEALING SCHEDULER
// ============================================================================
SchedulerResult RunDynamicWorkStealing(
    std::vector<Particle>& particles,
    size_t chunkSize,
    uint32_t numThreads,
    float dt,
    bool recordEvents
) {
    const size_t totalEntities = particles.size();
    const size_t numChunks = (totalEntities + chunkSize - 1) / chunkSize;

    std::atomic<size_t> nextChunkIdx{0};
    std::atomic<size_t> eventCount{0};
    std::vector<ImpactEvent> eventBuffer;
    if (recordEvents) {
        eventBuffer.resize(totalEntities); // Upper bound
    }

    std::vector<double> threadExecutionTimes(numThreads, 0.0);
    std::vector<double> threadStartTimesUs(numThreads, 0.0);

    std::barrier syncBarrier(numThreads + 1);
    std::atomic<bool> startFlag{false};
    auto dispatchStart = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    for (uint32_t t = 0; t < numThreads; ++t) {
        workers.emplace_back([&, t]() {
            syncBarrier.arrive_and_wait(); // Wait for all threads to be created

            while (!startFlag.load(std::memory_order_acquire)) {
                #if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
                #endif
            }
            auto threadRealStart = std::chrono::high_resolution_clock::now();
            threadStartTimesUs[t] = std::chrono::duration<double, std::micro>(threadRealStart - dispatchStart).count();

            // Dynamic Atomic Work Stealing Loop
            while (true) {
                size_t chunk = nextChunkIdx.fetch_add(1, std::memory_order_relaxed);
                if (chunk >= numChunks) break;

                size_t start = chunk * chunkSize;
                size_t end = std::min(start + chunkSize, totalEntities);

                for (size_t i = start; i < end; ++i) {
                    bool impacted = UpdateParticle(particles[i], dt);
                    if (recordEvents && impacted) {
                        // Non-deterministic atomic append order
                        size_t outIdx = eventCount.fetch_add(1, std::memory_order_relaxed);
                        eventBuffer[outIdx] = {
                            static_cast<uint32_t>(i),
                            std::sqrt(particles[i].velX * particles[i].velX + particles[i].velY * particles[i].velY),
                            particles[i].posX, particles[i].posY, particles[i].posZ
                        };
                    }
                }
            }

            auto threadEnd = std::chrono::high_resolution_clock::now();
            threadExecutionTimes[t] = std::chrono::duration<double, std::milli>(threadEnd - threadRealStart).count();
        });
    }

    syncBarrier.arrive_and_wait();
    dispatchStart = std::chrono::high_resolution_clock::now();
    startFlag.store(true, std::memory_order_release);

    for (auto& w : workers) {
        w.join();
    }
    auto totalEnd = std::chrono::high_resolution_clock::now();

    SchedulerResult res;
    res.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - dispatchStart).count();
    res.dispatchLatencyUs = std::accumulate(threadStartTimesUs.begin(), threadStartTimesUs.end(), 0.0) / numThreads;

    res.minThreadTimeMs = *std::min_element(threadExecutionTimes.begin(), threadExecutionTimes.end());
    res.maxThreadTimeMs = *std::max_element(threadExecutionTimes.begin(), threadExecutionTimes.end());
    double meanTime = std::accumulate(threadExecutionTimes.begin(), threadExecutionTimes.end(), 0.0) / numThreads;
    double sqDiffSum = 0.0;
    for (double tTime : threadExecutionTimes) {
        sqDiffSum += (tTime - meanTime) * (tTime - meanTime);
    }
    res.threadJitterStdDevMs = std::sqrt(sqDiffSum / numThreads);
    res.throughputMeps = (totalEntities / (res.totalTimeMs / 1000.0)) / 1e6;

    res.stateHash = FNV1a_64(particles.data(), totalEntities * sizeof(Particle));
    res.totalEvents = eventCount.load();
    res.eventBufferHash = recordEvents ? FNV1a_64(eventBuffer.data(), res.totalEvents * sizeof(ImpactEvent)) : 0;

    return res;
}

// ============================================================================
// 2. PRE-PARTITIONED CHUNK RANGE SCHEDULER ([start, end])
// ============================================================================
SchedulerResult RunPrePartitionedRanges(
    std::vector<Particle>& particles,
    size_t chunkSize,
    uint32_t numThreads,
    float dt,
    bool recordEvents
) {
    const size_t totalEntities = particles.size();

    // Per-thread pre-allocated event storage to maintain 100% deterministic event ordering
    struct ThreadLocalEventStorage {
        std::vector<ImpactEvent> events;
    };
    std::vector<ThreadLocalEventStorage> threadEvents(numThreads);
    if (recordEvents) {
        for (auto& te : threadEvents) {
            te.events.reserve(totalEntities / numThreads + 1024);
        }
    }

    std::vector<double> threadExecutionTimes(numThreads, 0.0);
    std::vector<double> threadStartTimesUs(numThreads, 0.0);

    std::barrier syncBarrier(numThreads + 1);
    std::atomic<bool> startFlag{false};
    auto dispatchStart = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    for (uint32_t t = 0; t < numThreads; ++t) {
        workers.emplace_back([&, t]() {
            syncBarrier.arrive_and_wait();

            while (!startFlag.load(std::memory_order_acquire)) {
                #if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
                #endif
            }
            auto threadRealStart = std::chrono::high_resolution_clock::now();
            threadStartTimesUs[t] = std::chrono::duration<double, std::micro>(threadRealStart - dispatchStart).count();

            // Static Partitioning: Each thread gets an exact, disjoint [threadStart, threadEnd) range
            size_t itemsPerThread = totalEntities / numThreads;
            size_t threadStart = t * itemsPerThread;
            size_t threadEnd = (t == numThreads - 1) ? totalEntities : threadStart + itemsPerThread;

            // Process chunks sequentially within thread's partition
            for (size_t cStart = threadStart; cStart < threadEnd; cStart += chunkSize) {
                size_t cEnd = std::min(cStart + chunkSize, threadEnd);
                for (size_t i = cStart; i < cEnd; ++i) {
                    bool impacted = UpdateParticle(particles[i], dt);
                    if (recordEvents && impacted) {
                        threadEvents[t].events.push_back({
                            static_cast<uint32_t>(i),
                            std::sqrt(particles[i].velX * particles[i].velX + particles[i].velY * particles[i].velY),
                            particles[i].posX, particles[i].posY, particles[i].posZ
                        });
                    }
                }
            }

            auto threadEndT = std::chrono::high_resolution_clock::now();
            threadExecutionTimes[t] = std::chrono::duration<double, std::milli>(threadEndT - threadRealStart).count();
        });
    }

    syncBarrier.arrive_and_wait();
    dispatchStart = std::chrono::high_resolution_clock::now();
    startFlag.store(true, std::memory_order_release);

    for (auto& w : workers) {
        w.join();
    }
    auto totalEnd = std::chrono::high_resolution_clock::now();

    SchedulerResult res;
    res.totalTimeMs = std::chrono::duration<double, std::milli>(totalEnd - dispatchStart).count();
    res.dispatchLatencyUs = std::accumulate(threadStartTimesUs.begin(), threadStartTimesUs.end(), 0.0) / numThreads;

    res.minThreadTimeMs = *std::min_element(threadExecutionTimes.begin(), threadExecutionTimes.end());
    res.maxThreadTimeMs = *std::max_element(threadExecutionTimes.begin(), threadExecutionTimes.end());
    double meanTime = std::accumulate(threadExecutionTimes.begin(), threadExecutionTimes.end(), 0.0) / numThreads;
    double sqDiffSum = 0.0;
    for (double tTime : threadExecutionTimes) {
        sqDiffSum += (tTime - meanTime) * (tTime - meanTime);
    }
    res.threadJitterStdDevMs = std::sqrt(sqDiffSum / numThreads);
    res.throughputMeps = (totalEntities / (res.totalTimeMs / 1000.0)) / 1e6;

    res.stateHash = FNV1a_64(particles.data(), totalEntities * sizeof(Particle));

    // Deterministic linear assembly of partitioned events
    if (recordEvents) {
        size_t totalEv = 0;
        for (const auto& te : threadEvents) totalEv += te.events.size();
        res.totalEvents = totalEv;

        std::vector<ImpactEvent> consolidatedEvents;
        consolidatedEvents.reserve(totalEv);
        for (const auto& te : threadEvents) {
            consolidatedEvents.insert(consolidatedEvents.end(), te.events.begin(), te.events.end());
        }
        res.eventBufferHash = FNV1a_64(consolidatedEvents.data(), totalEv * sizeof(ImpactEvent));
    } else {
        res.totalEvents = 0;
        res.eventBufferHash = 0;
    }

    return res;
}

} // namespace Benchmark

int main(int argc, char** argv) {
    const uint32_t numThreads = std::thread::hardware_concurrency();
    std::cout << "================================================================================\n";
    std::cout << "TASK SCHEDULERS & CONCURRENCY BENCHMARK (Hardware Threads: " << numThreads << ")\n";
    std::cout << "================================================================================\n\n";

    const std::vector<size_t> entityCounts = { 100000, 500000, 1000000 };
    const std::vector<size_t> chunkSizes = { 256, 512, 1024, 4096 };
    const float dt = 0.0166667f;

    std::cout << "[PART 1] Chunk Size Scaling & Work-Stealing Jitter Analysis\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << std::left 
              << std::setw(10) << "Entities"
              << std::setw(8)  << "Chunk"
              << std::setw(14) << "Scheduler"
              << std::setw(12) << "Time (ms)"
              << std::setw(14) << "M.Ent/sec"
              << std::setw(15) << "Disp.Lat(us)"
              << std::setw(13) << "Jitter StdDev"
              << std::setw(13) << "Spread (ms)"
              << "\n--------------------------------------------------------------------------------\n";

    for (size_t entities : entityCounts) {
        for (size_t chunk : chunkSizes) {
            // Dynamic Work Stealing
            {
                std::vector<Benchmark::Particle> particles;
                Benchmark::InitParticles(particles, entities);
                auto res = Benchmark::RunDynamicWorkStealing(particles, chunk, numThreads, dt, false);
                double spread = res.maxThreadTimeMs - res.minThreadTimeMs;

                std::cout << std::left 
                          << std::setw(10) << entities
                          << std::setw(8)  << chunk
                          << std::setw(14) << "Dyn-Atomic"
                          << std::setw(12) << std::fixed << std::setprecision(3) << res.totalTimeMs
                          << std::setw(14) << std::fixed << std::setprecision(2) << res.throughputMeps
                          << std::setw(15) << std::fixed << std::setprecision(2) << res.dispatchLatencyUs
                          << std::setw(13) << std::fixed << std::setprecision(3) << res.threadJitterStdDevMs
                          << std::setw(13) << std::fixed << std::setprecision(3) << spread
                          << "\n";
            }
            // Pre-Partitioned Ranges
            {
                std::vector<Benchmark::Particle> particles;
                Benchmark::InitParticles(particles, entities);
                auto res = Benchmark::RunPrePartitionedRanges(particles, chunk, numThreads, dt, false);
                double spread = res.maxThreadTimeMs - res.minThreadTimeMs;

                std::cout << std::left 
                          << std::setw(10) << entities
                          << std::setw(8)  << chunk
                          << std::setw(14) << "Pre-Part[S,E]"
                          << std::setw(12) << std::fixed << std::setprecision(3) << res.totalTimeMs
                          << std::setw(14) << std::fixed << std::setprecision(2) << res.throughputMeps
                          << std::setw(15) << std::fixed << std::setprecision(2) << res.dispatchLatencyUs
                          << std::setw(13) << std::fixed << std::setprecision(3) << res.threadJitterStdDevMs
                          << std::setw(13) << std::fixed << std::setprecision(3) << spread
                          << "\n";
            }
        }
    }

    std::cout << "\n--------------------------------------------------------------------------------\n";
    std::cout << "[PART 2] Rigorous Determinism Test: Atomic Work-Stealing vs Pre-Partitioned Ranges\n";
    std::cout << "         (1,000,000 Entities, Chunk Size 1024, 10 Consecutive Independent Runs)\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    const size_t testEntities = 1000000;
    const size_t testChunk = 1024;
    const int numRuns = 10;

    std::vector<uint64_t> dynStateHashes(numRuns);
    std::vector<uint64_t> dynEventHashes(numRuns);
    std::vector<uint64_t> partStateHashes(numRuns);
    std::vector<uint64_t> partEventHashes(numRuns);

    std::cout << "\nExecuting 10 Runs for Dynamic Atomic Work-Stealing...\n";
    for (int r = 0; r < numRuns; ++r) {
        std::vector<Benchmark::Particle> particles;
        Benchmark::InitParticles(particles, testEntities);
        auto res = Benchmark::RunDynamicWorkStealing(particles, testChunk, numThreads, dt, true);
        dynStateHashes[r] = res.stateHash;
        dynEventHashes[r] = res.eventBufferHash;
        std::cout << "  Run " << std::setw(2) << (r + 1) << " -> State Hash: 0x" << std::hex << std::setw(16) << std::setfill('0')
                  << res.stateHash << " | Event Buffer Hash: 0x" << std::setw(16) << res.eventBufferHash 
                  << " (Events: " << std::dec << res.totalEvents << ")" << std::setfill(' ') << "\n";
    }

    std::cout << "\nExecuting 10 Runs for Pre-Partitioned Chunk Ranges...\n";
    for (int r = 0; r < numRuns; ++r) {
        std::vector<Benchmark::Particle> particles;
        Benchmark::InitParticles(particles, testEntities);
        auto res = Benchmark::RunPrePartitionedRanges(particles, testChunk, numThreads, dt, true);
        partStateHashes[r] = res.stateHash;
        partEventHashes[r] = res.eventBufferHash;
        std::cout << "  Run " << std::setw(2) << (r + 1) << " -> State Hash: 0x" << std::hex << std::setw(16) << std::setfill('0')
                  << res.stateHash << " | Event Buffer Hash: 0x" << std::setw(16) << res.eventBufferHash 
                  << " (Events: " << std::dec << res.totalEvents << ")" << std::setfill(' ') << "\n";
    }

    bool dynEventIdentical = std::all_of(dynEventHashes.begin(), dynEventHashes.end(), [&](uint64_t h) { return h == dynEventHashes[0]; });
    bool partEventIdentical = std::all_of(partEventHashes.begin(), partEventHashes.end(), [&](uint64_t h) { return h == partEventHashes[0]; });
    bool partStateIdentical = std::all_of(partStateHashes.begin(), partStateHashes.end(), [&](uint64_t h) { return h == partStateHashes[0]; });

    std::cout << "\n--------------------------------------------------------------------------------\n";
    std::cout << "DETERMINISM VERDICT:\n";
    std::cout << "  - Dynamic Atomic Work-Stealing Event Buffer Invariance: " 
              << (dynEventIdentical ? "PASS (Invariant)" : "FAIL (Non-Deterministic Divergence across runs!)") << "\n";
    std::cout << "  - Pre-Partitioned Chunk Ranges Event Buffer Invariance: " 
              << (partEventIdentical ? "PASS (100% Bit-Exact Identical across all 10 runs!)" : "FAIL") << "\n";
    std::cout << "  - Pre-Partitioned In-Place State Invariance:           " 
              << (partStateIdentical ? "PASS (100% Bit-Exact Identical across all 10 runs!)" : "FAIL") << "\n";
    std::cout << "================================================================================\n";

    return 0;
}
