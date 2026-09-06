#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <cstdint>
#include <type_traits>

// ============================================================================
// Bitsery Mock Header Headers
// ============================================================================
namespace bitsery {
    struct OutputAdapter {
        std::vector<uint8_t> buffer;
        template<typename T> void value4b(T val) {
            uint8_t bytes[4];
            std::memcpy(bytes, &val, 4);
            buffer.insert(buffer.end(), bytes, bytes + 4);
        }
    };
}

// ============================================================================
// 1. Core ECS Component Definitions (Clean, Aligned, Zero-Macro Noise)
// ============================================================================

struct alignas(16) TransformComponent {
    float posX{10.0f}, posY{20.0f}, posZ{30.0f}, posW{1.0f};
    float scaleX{1.0f}, scaleY{1.0f}, scaleZ{1.0f}, scaleW{1.0f};
};

struct alignas(16) RigidBodyComponent {
    float velX{0.5f}, velY{1.2f}, velZ{-0.3f}, mass{75.0f};
    uint32_t flags{0x1};
    uint32_t entityID{101};
};

// ============================================================================
// 2. Dual-Purpose Visitor Schema (Binds Bitsery + Reflection in 1 Function)
// ============================================================================

template <typename Target, typename Visitor>
void VisitComponent(Target& c, Visitor&& visitor) {
    if constexpr (std::is_same_v<std::decay_t<Target>, TransformComponent>) {
        visitor("Position X", c.posX);
        visitor("Position Y", c.posY);
        visitor("Position Z", c.posZ);
        visitor("Position W", c.posW);
        visitor("Scale X", c.scaleX);
        visitor("Scale Y", c.scaleY);
        visitor("Scale Z", c.scaleZ);
        visitor("Scale W", c.scaleW);
    } else if constexpr (std::is_same_v<std::decay_t<Target>, RigidBodyComponent>) {
        visitor("Velocity X", c.velX);
        visitor("Velocity Y", c.velY);
        visitor("Velocity Z", c.velZ);
        visitor("Mass", c.mass);
        visitor("Flags", c.flags);
        visitor("Entity ID", c.entityID);
    }
}

// ============================================================================
// 3. Bitsery Binding Adaptor (Consumes Dual-Purpose Visitor)
// ============================================================================

template <typename S, typename T>
void serialize(S& s, T& component) {
    VisitComponent(component, [&](const char* name, auto& value) {
        using V = std::decay_t<decltype(value)>;
        if constexpr (sizeof(V) == 4) {
            s.value4b(value);
        }
    });
}

// ============================================================================
// 4. ImGui GUI Property Inspector (Consumes Dual-Purpose Visitor)
// ============================================================================

struct MockImGui {
    static void DragFloat(const char* label, float* val) {
        std::cout << "  [ImGui::DragFloat] " << label << " = " << *val << "\n";
    }
    static void InputUint32(const char* label, uint32_t* val) {
        std::cout << "  [ImGui::InputUint32] " << label << " = " << *val << "\n";
    }
};

template <typename T>
void RenderGenericImGuiInspector(const char* componentName, T& component) {
    std::cout << "=== ImGui Inspector Window: [" << componentName << "] ===\n";
    VisitComponent(component, [](const char* name, auto& value) {
        using V = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<V, float>) {
            MockImGui::DragFloat(name, &value);
        } else if constexpr (std::is_same_v<V, uint32_t>) {
            MockImGui::InputUint32(name, &value);
        }
    });
    std::cout << "\n";
}

// ============================================================================
// Main Proof-of-Concept Execution
// ============================================================================

int main() {
    std::cout << "==========================================================================\n";
    std::cout << "  PROOF-OF-CONCEPT: DUAL-PURPOSE BITSERY + IMGUI VISITOR PATTERN\n";
    std::cout << "==========================================================================\n\n";

    TransformComponent transform;
    RigidBodyComponent body;

    // 1. Render ImGui Editor Property Inspection
    RenderGenericImGuiInspector("Transform Component", transform);
    RenderGenericImGuiInspector("RigidBody Component", body);

    // 2. Serialize exact same component using Bitsery Output Adapter
    bitsery::OutputAdapter bitseryStream;
    serialize(bitseryStream, transform);
    serialize(bitseryStream, body);

    std::cout << "=== Bitsery Binary Serialization Output ===\n";
    std::cout << "Successfully serialized Transform + RigidBody to " << bitseryStream.buffer.size() << " bytes.\n";
    std::cout << "First 16 raw bytes: 0x";
    for (size_t i = 0; i < 16 && i < bitseryStream.buffer.size(); ++i) {
        std::cout << std::hex << (int)bitseryStream.buffer[i] << " ";
    }
    std::cout << std::dec << "\n\n";
    std::cout << "VERDICT: Dual-visitor architecture seamlessly unifies ImGui and Bitsery!\n";

    return 0;
}
