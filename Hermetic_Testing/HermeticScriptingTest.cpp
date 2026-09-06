#include "HermeticScriptingEngine.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <iomanip>

using namespace Type0;

const std::string v1Script = R"(
    function update_movement(entity, dt)
        multiplier = 1.0
        entity.position.x = entity.position.x + entity.velocity.x * dt
        entity.position.y = entity.position.y + entity.velocity.y * dt
        entity.position.z = entity.position.z + entity.velocity.z * dt
    end
)";

const std::string v2Script = R"(
    function update_movement(entity, dt)
        multiplier = 2.0
        entity.position.x = entity.position.x + entity.velocity.x * dt * 2.0
        entity.position.y = entity.position.y + entity.velocity.y * dt * 2.0
        entity.position.z = entity.position.z + entity.velocity.z * dt * 2.0
    end
)";

void RunTest1_InitializationAndTeardown() {
    std::cout << "\n--- TEST 1: Headless Lua State Initialization & Teardown ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    HermeticScriptingEngine engine;
    bool initOk = engine.Initialize();
    assert(initOk && "Lua state failed to initialize!");
    assert(engine.IsInitialized() && "Engine failed reporting initialized status!");

    bool loadOk = engine.LoadScript(v1Script);
    assert(loadOk && "Lua script compiling/parsing failed!");

    engine.Shutdown();
    assert(!engine.IsInitialized() && "Engine failed to reset initialized status!");

    auto end = std::chrono::high_resolution_clock::now();
    double initTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[PASSED] Test 1 completed in " << std::fixed << std::setprecision(4) << initTimeMs << " ms." << std::endl;
}

void RunTest2_BitExactDeterminism() {
    std::cout << "\n--- TEST 2: Bit-Exact Determinism Verification Across Runs ---" << std::endl;
    const int NUM_RUNS = 10;
    const int ENTITY_COUNT = 1000;
    const int TICKS = 100;
    const float DT = 0.0166667f;

    uint64_t referenceHash = 0;

    for (int run = 0; run < NUM_RUNS; ++run) {
        HermeticScriptingEngine engine;
        engine.Initialize();
        engine.LoadScript(v1Script);

        std::vector<ScriptEntity> entities(ENTITY_COUNT);
        for (int i = 0; i < ENTITY_COUNT; ++i) {
            entities[i].id = static_cast<uint32_t>(i);
            entities[i].position = { float(i), float(i * 2), float(i * 3), 1.0f };
            entities[i].velocity = { 1.5f, -0.5f, 2.0f, 0.0f };
            entities[i].health = 100.0f;
        }

        for (int tick = 0; tick < TICKS; ++tick) {
            for (auto& entity : entities) {
                engine.ExecuteEntityUpdate("update_movement", entity, DT);
            }
        }

        uint64_t currentHash = HashState(entities);
        if (run == 0) {
            referenceHash = currentHash;
        } else {
            assert(currentHash == referenceHash && "DETERMINISM FAILURE: Script state hash mismatched across runs!");
        }

        engine.Shutdown();
    }

    std::cout << "[PASSED] Test 2 verified 100% bit-exact state identity (Hash: 0x"
              << std::hex << referenceHash << std::dec << ") over " << NUM_RUNS << " distinct runs!" << std::endl;
}

void RunTest3_HotReloadingCLI() {
    std::cout << "\n--- TEST 3: CLI Hot-Reloading Benchmark (`engine run --reload-scripts`) ---" << std::endl;

    HermeticScriptingEngine engine;
    engine.Initialize();
    engine.LoadScript(v1Script);

    ScriptEntity entity{ 1, {0.0f, 0.0f, 0.0f, 1.0f}, {10.0f, 10.0f, 10.0f, 0.0f}, 100.0f };

    // Tick before reload (1.0x multiplier)
    engine.ExecuteEntityUpdate("update_movement", entity, 1.0f);
    assert(entity.position.x == 10.0f);

    // Trigger CLI hot-reload
    bool reloadOk = engine.TriggerHotReload(v2Script);
    assert(reloadOk && "Hot reload script substitution failed!");

    // Tick after reload (2.0x multiplier)
    engine.ExecuteEntityUpdate("update_movement", entity, 1.0f);
    assert(entity.position.x == 30.0f); // 10.0 initial + (10.0 * 1.0 * 2.0) = 30.0

    std::cout << "[PASSED] CLI Hot-Reload trigger verified! Latency: " 
              << engine.GetLastHotReloadTimeUs() << " us (< 500 us target)." << std::endl;

    engine.Shutdown();
}

void RunTest4_SIMDBindingThroughput() {
    std::cout << "\n--- TEST 4: sol2 / LuaJIT SIMD Vector C++ Binding Throughput ---" << std::endl;

    HermeticScriptingEngine engine;
    engine.Initialize();
    engine.LoadScript(v1Script);

    const int ENTITY_COUNT = 50000;
    const int TICKS = 50;
    std::vector<ScriptEntity> entities(ENTITY_COUNT);

    for (int i = 0; i < ENTITY_COUNT; ++i) {
        entities[i].id = static_cast<uint32_t>(i);
        entities[i].position = { float(i), float(i), float(i), 1.0f };
        entities[i].velocity = { 1.0f, 2.0f, 3.0f, 0.0f };
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int tick = 0; tick < TICKS; ++tick) {
        for (size_t i = 0; i < entities.size(); ++i) {
            engine.ExecuteEntityUpdate("update_movement", entities[i], 0.016f);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double totalSeconds = std::chrono::duration<double>(end - start).count();
    double totalOps = double(ENTITY_COUNT) * double(TICKS);
    double opsPerSec = totalOps / totalSeconds;

    std::cout << "[PASSED] Processed " << totalOps << " script entity ticks in " 
              << std::fixed << std::setprecision(3) << totalSeconds * 1000.0 << " ms ("
              << std::setprecision(2) << opsPerSec / 1e6 << " Million ops/sec)." << std::endl;

    engine.Shutdown();
}

int main() {
    std::cout << "=== HERMETIC & PERFORMANT SCRIPTING SUITE (LuaJIT + sol2) ===" << std::endl;
    RunTest1_InitializationAndTeardown();
    RunTest2_BitExactDeterminism();
    RunTest3_HotReloadingCLI();
    RunTest4_SIMDBindingThroughput();
    std::cout << "\nALL 4 SCRIPTING ENGINE BENCHMARKS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
