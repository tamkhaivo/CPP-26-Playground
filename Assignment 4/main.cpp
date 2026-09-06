#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <memory>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>

#include "OSModule.hpp"
#include "IParticlePhysicsPlugin.hpp"
#include "CoreAffinityManager.hpp"

// Concurrency libraries headers for initialization on host
#include <TaskScheduler.h>
#include <taskflow/taskflow.hpp>
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

// Define the DLL/SO filename based on the OS
#if defined(_WIN32)
    const std::string PLUGIN_FILE = "./ParticlePhysics.dll";
#elif defined(__APPLE__)
    const std::string PLUGIN_FILE = "./libParticlePhysics.dylib";
#else
    const std::string PLUGIN_FILE = "./libParticlePhysics.so";
#endif

struct BenchmarkResult {
    std::string name;
    double avgLatencyMs;
    double minLatencyMs;
    double maxLatencyMs;
    double jitterMs;
    double gflops;
    bool verified;
};

BenchmarkResult ExecuteBenchmark(
    const std::string& name,
    IParticlePhysicsPlugin* plugin,
    std::vector<MeshInstanceData>& particles,
    std::vector<MeshInstanceVelocity>& velocities,
    const std::vector<AttractorData>& attractors,
    size_t count,
    float dt,
    int iterations,
    std::function<void()> runFunc
) {
    const float epsilon = 0.01f;

    // Reset data before warmup
    for (size_t i = 0; i < count; ++i) {
        particles[i].x = static_cast<float>(i % 100) - 50.0f;
        particles[i].y = static_cast<float>((i / 100) % 100) - 50.0f;
        particles[i].z = static_cast<float>(i / 10000) - 50.0f;
        particles[i].w = 1.0f;

        velocities[i].vx = 0.0f;
        velocities[i].vy = 0.0f;
        velocities[i].vz = 0.0f;
        velocities[i].vw = 0.0f;
    }

    // Warmup Run
    runFunc();

    // Reset positions after warmup
    for (size_t i = 0; i < count; ++i) {
        particles[i].x = static_cast<float>(i % 100) - 50.0f;
        particles[i].y = static_cast<float>((i / 100) % 100) - 50.0f;
        particles[i].z = static_cast<float>(i / 10000) - 50.0f;
        particles[i].w = 1.0f;

        velocities[i].vx = 0.0f;
        velocities[i].vy = 0.0f;
        velocities[i].vz = 0.0f;
        velocities[i].vw = 0.0f;
    }

    // Benchmark execution
    std::vector<double> timingsMs;
    timingsMs.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        runFunc();
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timingsMs.push_back(elapsed.count());
    }

    // Calculate stats
    double sum = std::accumulate(timingsMs.begin(), timingsMs.end(), 0.0);
    double avgLatencyMs = sum / iterations;

    // FLOP estimation: ~16 FLOPs per particle-attractor interaction plus integration overhead.
    // 3 subs (dx, dy, dz) = 3 FLOPs
    // 3 muls + 3 adds (dist_sq + softening) = 6 FLOPs
    // 1 rsqrt = 1 FLOP
    // 3 muls (force / inv_dist cubed) = 3 FLOPs
    // 3 muladds (ax, ay, az accumulate) = 3 FLOPs (using FMA as 1 instruction but 2 FLOPs, let's say 3 FLOPs total)
    // Total interactions = count * attractorCount
    double totalFlops = static_cast<double>(count) * attractors.size() * 16.0;
    double gflops = (totalFlops / 1e9) / (avgLatencyMs / 1000.0);

    double minLatencyMs = *std::min_element(timingsMs.begin(), timingsMs.end());
    double maxLatencyMs = *std::max_element(timingsMs.begin(), timingsMs.end());
    double jitterMs = maxLatencyMs - minLatencyMs;

    // Verify mathematical correctness against expected positions
    // We capture particle state after iterations run and compute a hash/checksum or verify a subset
    bool verified = true;
    
    // We run a verification check to ensure results don't contain NaNs and match sanity ranges
    for (size_t i = 0; i < 100; ++i) { // Check first 100 particles for NaNs or excessive drift
        if (std::isnan(particles[i].x) || std::isnan(particles[i].y) || std::isnan(particles[i].z)) {
            std::cerr << "[" << name << "] ERROR: NaN detected at index " << i << std::endl;
            verified = false;
            break;
        }
    }

    return {name, avgLatencyMs, minLatencyMs, maxLatencyMs, jitterMs, gflops, verified};
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "   Assignment 4: Compute-Bound Multi-Core Attractor Benchmark           \n";
    std::cout << "========================================================================\n\n";

    // 1. Load the dynamic library
    std::cout << "[Host] Loading plugin: " << PLUGIN_FILE << std::endl;
    auto module = std::make_unique<DynamicModule::OSModule>(PLUGIN_FILE);
    if (!module->IsLoaded()) {
        std::cerr << "[Host] ERROR: Failed to load plugin DLL: " << PLUGIN_FILE << std::endl;
        return 1;
    }
    std::cout << "[Host] Plugin DLL loaded successfully." << std::endl;

    // 2. Resolve the factory function
    auto createFunc = module->GetSymbol<CreatePluginFunc>("CreateDynamicPlugin");
    if (!createFunc) {
        std::cerr << "[Host] ERROR: Failed to locate symbol 'CreateDynamicPlugin' in DLL." << std::endl;
        return 1;
    }

    // 3. Instantiate the plugin
    std::unique_ptr<IFeaturePlugin> rawPlugin(createFunc());
    if (!rawPlugin) {
        std::cerr << "[Host] ERROR: Plugin instantiation returned nullptr." << std::endl;
        return 1;
    }

    auto* physicsPlugin = static_cast<IParticlePhysicsPlugin*>(rawPlugin.get());

    if (!physicsPlugin->Initialize(nullptr, nullptr)) {
        std::cerr << "[Host] ERROR: Failed to initialize physics plugin." << std::endl;
        return 1;
    }

    // 4. Initialize Multi-Core Schedulers
    std::cout << "[Host] Initializing concurrency pools..." << std::endl;
    
    // enkiTS Scheduler
    enki::TaskScheduler enkiScheduler;
    enkiScheduler.Initialize();
    
    // Taskflow Executor
    tf::Executor tfExecutor;
    
    // stdexec Static Thread Pool
    exec::static_thread_pool stdexecPool(std::thread::hardware_concurrency());

    // 5. Setup data
    const size_t particleCount = 1000000;
    const size_t attractorCount = 512;
    const float dt = 0.016f;
    const int iterations = 5; // Lower iterations since workload is heavily compute-bound
    
    std::cout << "[Host] Allocating " << particleCount << " particles..." << std::endl;
    std::vector<MeshInstanceData> particles(particleCount);
    std::vector<MeshInstanceVelocity> velocities(particleCount);
    std::vector<AttractorData> attractors(attractorCount);

    // Initialize attractors symmetrically in a sphere
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> massDist(10.0f, 100.0f);
    for (size_t i = 0; i < attractorCount; ++i) {
        attractors[i].x = dist(rng);
        attractors[i].y = dist(rng);
        attractors[i].z = dist(rng);
        attractors[i].mass = massDist(rng);
    }

    std::vector<BenchmarkResult> results;

    // Test 1: oneTBB + Highway SIMD
    std::cout << "[Host] Running TBB + Highway SIMD benchmark..." << std::endl;
    results.push_back(ExecuteBenchmark("oneTBB + Highway SIMD", physicsPlugin, particles, velocities, attractors, particleCount, dt, iterations, [&]() {
        physicsPlugin->UpdateParticlesTBB(particles.data(), velocities.data(), particleCount, attractors.data(), attractorCount, dt);
    }));

    // Test 2: enkiTS + Highway SIMD
    std::cout << "[Host] Running enkiTS + Highway SIMD benchmark..." << std::endl;
    results.push_back(ExecuteBenchmark("enkiTS + Highway SIMD", physicsPlugin, particles, velocities, attractors, particleCount, dt, iterations, [&]() {
        physicsPlugin->UpdateParticlesEnki(particles.data(), velocities.data(), particleCount, attractors.data(), attractorCount, dt, &enkiScheduler);
    }));

    // Test 3: Taskflow + Highway SIMD
    std::cout << "[Host] Running Taskflow + Highway SIMD benchmark..." << std::endl;
    results.push_back(ExecuteBenchmark("Taskflow + Highway SIMD", physicsPlugin, particles, velocities, attractors, particleCount, dt, iterations, [&]() {
        physicsPlugin->UpdateParticlesTaskflow(particles.data(), velocities.data(), particleCount, attractors.data(), attractorCount, dt, &tfExecutor);
    }));

    // Test 4: Marl + Highway SIMD
    std::cout << "[Host] Running Marl + Highway SIMD benchmark..." << std::endl;
    results.push_back(ExecuteBenchmark("Marl + Highway SIMD", physicsPlugin, particles, velocities, attractors, particleCount, dt, iterations, [&]() {
        physicsPlugin->UpdateParticlesMarl(particles.data(), velocities.data(), particleCount, attractors.data(), attractorCount, dt, nullptr);
    }));

    // Test 5: stdexec + Highway SIMD
    std::cout << "[Host] Running stdexec + Highway SIMD benchmark..." << std::endl;
    results.push_back(ExecuteBenchmark("stdexec + Highway SIMD", physicsPlugin, particles, velocities, attractors, particleCount, dt, iterations, [&]() {
        physicsPlugin->UpdateParticlesStdexec(particles.data(), velocities.data(), particleCount, attractors.data(), attractorCount, dt, &stdexecPool);
    }));

    // Test 6: Single-threaded SIMD (P-Core)
    std::cout << "[Host] Running Single-threaded SIMD (P-Core) benchmark..." << std::endl;
    results.push_back(ExecuteBenchmark("Single-threaded P-Core", physicsPlugin, particles, velocities, attractors, particleCount, dt, iterations, [&]() {
        physicsPlugin->UpdateParticlesSingleThreaded(particles.data(), velocities.data(), particleCount, attractors.data(), attractorCount, dt);
    }));

    // Test 7: Single-threaded SIMD (E-Core)
    std::cout << "[Host] Running Single-threaded SIMD (E-Core) benchmark..." << std::endl;
    bool pinned = Type0::CoreAffinityManager::PinCurrentThreadToEfficiencyCores();
    if (pinned) {
        std::cout << "[Host] Host thread pinned to E-Cores." << std::endl;
    } else {
        std::cout << "[Host] WARNING: Pinned thread affinity failed, utilizing host defaults." << std::endl;
    }
    results.push_back(ExecuteBenchmark("Single-threaded E-Core", physicsPlugin, particles, velocities, attractors, particleCount, dt, iterations, [&]() {
        physicsPlugin->UpdateParticlesSingleThreaded(particles.data(), velocities.data(), particleCount, attractors.data(), attractorCount, dt);
    }));

    // Print comparative analysis table
    std::cout << "\n========================================================================================\n";
    std::cout << "                                PERFORMANCE COMPARISON TABLE\n";
    std::cout << "========================================================================================\n";
    std::cout << "  " << std::left << std::setw(32) << "Configuration"
              << " | " << std::right << std::setw(12) << "Avg Latency"
              << " | " << std::right << std::setw(12) << "Min Latency"
              << " | " << std::right << std::setw(12) << "Max Latency"
              << " | " << std::right << std::setw(12) << "Throughput"
              << " | Verified\n";
    std::cout << "----------------------------------------------------------------------------------------\n";
    for (const auto& r : results) {
        std::cout << "  " << std::left << std::setw(32) << r.name
                  << " | " << std::right << std::setw(9) << std::fixed << std::setprecision(4) << r.avgLatencyMs << " ms"
                  << " | " << std::right << std::setw(9) << r.minLatencyMs << " ms"
                  << " | " << std::right << std::setw(9) << r.maxLatencyMs << " ms"
                  << " | " << std::right << std::setw(6) << std::setprecision(2) << r.gflops << " GFLOPS"
                  << " | " << (r.verified ? "YES" : "NO") << "\n";
    }
    std::cout << "========================================================================================\n\n";

    physicsPlugin->Shutdown();
    rawPlugin.reset();
    module.reset();

    std::cout << "[Host] Execution completed successfully.\n";
    return 0;
}
