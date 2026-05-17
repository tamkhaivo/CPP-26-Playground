#pragma once

// Forward declare Vulkan handles so this compiles without the SDK
typedef struct VkInstance_T* VkInstance;
typedef struct VkDevice_T* VkDevice;

struct IFeaturePlugin {
    virtual ~IFeaturePlugin() = default;
    virtual bool Initialize(VkInstance instance, VkDevice device) = 0;
    virtual void Shutdown() = 0;
    virtual void Update() = 0;
};

// Define the function pointer type for extracting the C-linkage export from the DLL
typedef IFeaturePlugin* (*CreatePluginFunc)();
