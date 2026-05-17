#include <iostream>
#include <memory>
#include "OSModule.hpp"
#include "IFeaturePlugin.hpp"
#include "PureVulkanFallback.hpp"


int main() {
    DynamicModule::OSModule nvidiaPlugin("DynamicLoading_NvidiaStreamline.dll");
    
    // We will store whichever plugin we end up creating here
    std::unique_ptr<IFeaturePlugin> plugin = nullptr;

    if (nvidiaPlugin.IsLoaded()) {
        std::cout << "NVIDIA Plugin loaded successfully. Extracting symbols..." << std::endl;
        
        // Extract the function pointer from the dynamically loaded DLL
        auto createFunc = nvidiaPlugin.GetSymbol<CreatePluginFunc>("CreateDynamicPlugin");
        
        if (createFunc) {
            plugin.reset(createFunc());
        } else {
            std::cout << "Failed to find 'CreateDynamicPlugin' inside the DLL." << std::endl;
        }
        
    } else {
        std::cout << "NVIDIA Plugin not found. Falling back..." << std::endl;
        // Use our static fallback compiled directly into the executable
        plugin.reset(CreateFallbackPlugin());
    }

    // Simulate the engine loop, completely oblivious to whether the DLL or fallback is running!
    if (plugin && plugin->Initialize(nullptr, nullptr)) {
        plugin->Update();
        plugin->Shutdown();
    }

    return 0;
}