#pragma once
#include <iostream>
#include "IFeaturePlugin.hpp"

class PureVulkanFallback : public IFeaturePlugin {
public:
    bool Initialize(VkInstance instance, VkDevice device) override {
        std::cout << "PureVulkanFallback::Initialize() - Falling back to pure Vulkan." << std::endl;
        return true;
    }

    void Shutdown() override {
        std::cout << "PureVulkanFallback::Shutdown()" << std::endl;
    }

    void Update() override {
        std::cout << "PureVulkanFallback::Update() - Running generic simulation." << std::endl;
    }
};

// Renamed from CreateType0Plugin to CreateFallbackPlugin to differentiate it from the DLL export
IFeaturePlugin* CreateFallbackPlugin() {
    return new PureVulkanFallback();
}
