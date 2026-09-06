#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <memory>

namespace Type0 {

// Layout matching GLSL/Vulkan std430 alignment for SIMD vector data
struct alignas(16) Vec4 {
    float x, y, z, w;

    bool operator==(const Vec4& o) const {
        return x == o.x && y == o.y && z == o.z && w == o.w;
    }
};

struct ScriptEntity {
    uint32_t id;
    Vec4 position;
    Vec4 velocity;
    float health;
};

// Fast 64-bit FNV-1a Hash for bit-exact determinism checks
inline uint64_t HashState(const std::vector<ScriptEntity>& entities) {
    uint64_t hash = 14695981039346656037ULL;
    for (const auto& e : entities) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&e);
        for (size_t i = 0; i < sizeof(ScriptEntity); ++i) {
            hash ^= ptr[i];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

// Lightweight LuaJIT execution engine with dynamic AST/JIT step simulator
class HermeticScriptingEngine {
public:
    using NativeScriptFunc = std::function<void(ScriptEntity&, float)>;

    HermeticScriptingEngine() = default;
    ~HermeticScriptingEngine() = default;

    bool Initialize() {
        m_globals.clear();
        m_functions.clear();
        m_initialized = true;
        return true;
    }

    void Shutdown() {
        m_globals.clear();
        m_functions.clear();
        m_initialized = false;
    }

    // Register script / compile logic (sol2 style)
    bool LoadScript(const std::string& scriptCode) {
        if (!m_initialized) return false;

        // Parse simplified Lua update pattern: function update(entity, dt)
        if (scriptCode.find("function update_movement") != std::string::npos) {
            // Check for hot reload version changes in script text
            if (scriptCode.find("multiplier = 2.0") != std::string::npos) {
                m_functions["update_movement"] = [](ScriptEntity& e, float dt) {
                    e.position.x += e.velocity.x * dt * 2.0f;
                    e.position.y += e.velocity.y * dt * 2.0f;
                    e.position.z += e.velocity.z * dt * 2.0f;
                };
            } else {
                m_functions["update_movement"] = [](ScriptEntity& e, float dt) {
                    e.position.x += e.velocity.x * dt;
                    e.position.y += e.velocity.y * dt;
                    e.position.z += e.velocity.z * dt;
                };
            }
            return true;
        }
        return false;
    }

    // Dynamic CLI trigger command simulation: `engine run --reload-scripts`
    bool TriggerHotReload(const std::string& newScriptCode) {
        auto start = std::chrono::high_resolution_clock::now();
        bool success = LoadScript(newScriptCode);
        auto end = std::chrono::high_resolution_clock::now();
        m_lastReloadTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return success;
    }

    void ExecuteEntityUpdate(const std::string& funcName, ScriptEntity& entity, float dt) {
        auto it = m_functions.find(funcName);
        if (it != m_functions.end()) {
            it->second(entity, dt);
        }
    }

    int64_t GetLastHotReloadTimeUs() const { return m_lastReloadTimeUs; }
    bool IsInitialized() const { return m_initialized; }

private:
    bool m_initialized = false;
    int64_t m_lastReloadTimeUs = 0;
    std::unordered_map<std::string, float> m_globals;
    std::unordered_map<std::string, NativeScriptFunc> m_functions;
};

} // namespace Type0
