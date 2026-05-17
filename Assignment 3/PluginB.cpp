#include <iostream>
#include "IFeaturePlugin.hpp"

class PluginB : public IFeaturePlugin {
public:
    bool Initialize(VkInstance instance, VkDevice device) override {
        std::cout << "[Plugin B] Physics Subsystem Initialized." << std::endl;
        return true;
    }
    void Shutdown() override { std::cout << "[Plugin B] Physics Subsystem Shutdown." << std::endl; }
    void Update() override { std::cout << "[Plugin B] Resolving collision constraints." << std::endl; }
};

extern "C" __declspec(dllexport) IFeaturePlugin* CreateDynamicPlugin() {
    return new PluginB();
}
