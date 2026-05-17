#pragma once
#include <vector>
#include <memory>
#include <filesystem>
#include <iostream>
#include "OSModule.hpp"
#include "IFeaturePlugin.hpp"

namespace DynamicModule {

class PluginManager {
public:
    void LoadPluginsFromDirectory(const std::string& directoryPath) {
        std::cout << "Scanning directory: " << directoryPath << std::endl;
        
        // TODO: Step 1. Iterate over all files in `directoryPath` using std::filesystem::directory_iterator
        std::filesystem::directory_iterator it(directoryPath);

        // TODO: Step 2. Check if the file is a regular file and its extension is ".dll" (or ".so")
        for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
            if (entry.is_regular_file() && (entry.path().extension() == ".dll" || entry.path().extension() == ".so" || entry.path().extension() == ".dylib")) {
                // TODO: Step 3. If it is a DLL, instantiate an OSModule using the file's absolute path
                // FIX: Allocate it directly via unique_ptr. If we create it on the stack, it will get destroyed at the end of the loop, unloading the DLL!
                auto module = std::make_unique<OSModule>(entry.path().string());
                
                // TODO: Step 4. If the OSModule IsLoaded(), GetSymbol<CreatePluginFunc>("CreateDynamicPlugin")
                if (module->IsLoaded()) {
                    auto createPlugin = module->GetSymbol<CreatePluginFunc>("CreateDynamicPlugin");
                    
                    // TODO: Step 5. If the symbol exists, invoke it and push the resulting IFeaturePlugin* into m_activePlugins
                    if (createPlugin) {
                        // FIX: m_activePlugins holds unique_ptrs, so we must wrap the raw pointer returned by createPlugin()
                        m_activePlugins.push_back(std::unique_ptr<IFeaturePlugin>(createPlugin()));
                        
                        // TODO: Step 6. Push the OSModule into m_loadedModules so the DLL isn't unloaded from memory!
                        // FIX: We std::move the unique pointer into our vector to safely transfer ownership
                        m_loadedModules.push_back(std::move(module));
                    }
                }
            }
        }
    }

    void InitializeAll() {
        for (auto& plugin : m_activePlugins) {
            plugin->Initialize(nullptr, nullptr);
        }
    }

    void UpdateAll() {
        for (auto& plugin : m_activePlugins) {
            plugin->Update();
        }
    }

    void ShutdownAll() {
        for (auto& plugin : m_activePlugins) {
            plugin->Shutdown();
        }
    }

private:
    std::vector<std::unique_ptr<OSModule>> m_loadedModules;
    std::vector<std::unique_ptr<IFeaturePlugin>> m_activePlugins;
};

} // namespace DynamicModule
