#include <iostream>
#include "PluginManager.hpp"

int main() {
    std::cout << "--- Assignment 3: Directory-Based Plugin Ingestion ---" << std::endl;
    
    DynamicModule::PluginManager pluginManager;
    
    // We point the manager to the plugins subdirectory which CMake automatically populated for us
    pluginManager.LoadPluginsFromDirectory("build/plugins/Debug");

    std::cout << "\n--- Simulating Engine Loop ---" << std::endl;
    pluginManager.InitializeAll();
    pluginManager.UpdateAll();
    pluginManager.ShutdownAll();

    return 0;
}
