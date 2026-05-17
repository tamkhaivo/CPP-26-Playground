#include <iostream>
#include "IFeaturePlugin.hpp"

class NvidiaStreamlinePlugin : public IFeaturePlugin {
public:
    bool Initialize(VkInstance instance, VkDevice device) override {
        std::cout << "[DLL] NvidiaStreamlinePlugin::Initialize() - Hooking into Vulkan with DLSS/FrameGen!" << std::endl;
        return true;
    }

    void Shutdown() override {
        std::cout << "[DLL] NvidiaStreamlinePlugin::Shutdown()" << std::endl;
    }

    void Update() override {
        std::cout << "[DLL] NvidiaStreamlinePlugin::Update() - Processing DLSS frame data." << std::endl;
    }
};

// This is the ABI boundary!
// By exporting this as extern "C", we avoid C++ name mangling, allowing the engine to find exactly "CreateType0Plugin".
extern "C" __declspec(dllexport) IFeaturePlugin* CreateDynamicPlugin() {
    return new NvidiaStreamlinePlugin();
}
