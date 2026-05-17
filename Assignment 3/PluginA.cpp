#include <iostream>
#include "IFeaturePlugin.hpp"

class PluginA : public IFeaturePlugin {
public:
    bool Initialize(VkInstance instance, VkDevice device) override {
        std::cout << "[Plugin A] Audio System Initialized." << std::endl;
        return true;
    }
    void Shutdown() override { std::cout << "[Plugin A] Audio System Shutdown." << std::endl; }
    void Update() override { std::cout << "[Plugin A] Processing spatial audio." << std::endl; }
};

extern "C" __declspec(dllexport) IFeaturePlugin* CreateDynamicPlugin() {
    return new PluginA();
}
