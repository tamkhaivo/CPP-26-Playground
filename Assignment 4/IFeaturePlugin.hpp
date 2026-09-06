#pragma once

// Forward declare Vulkan handles to avoid build-time dependencies
typedef struct VkInstance_T* VkInstance;
typedef struct VkDevice_T* VkDevice;

struct IFeaturePlugin {
    virtual ~IFeaturePlugin() = default;
    virtual bool Initialize(VkInstance instance, VkDevice device) = 0;
    virtual void Shutdown() = 0;
    virtual void Update() = 0;
};

// Function pointer signature for DLL entry point
typedef IFeaturePlugin* (*CreatePluginFunc)();
