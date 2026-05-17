#pragma once
#include <string>
#include <string_view>

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace DynamicModule {

class OSModule {
public:
    explicit OSModule(std::string_view moduleName) {
#if defined(_WIN32)
        m_handle = LoadLibraryA(moduleName.data());
#else
        m_handle = dlopen(moduleName.data(), RTLD_NOW);
#endif
    }

    ~OSModule() {
        if (m_handle) {
#if defined(_WIN32)
            FreeLibrary(static_cast<HMODULE>(m_handle));
#else
            dlclose(m_handle);
#endif
        }
    }

    [[nodiscard]] bool IsLoaded() const { return m_handle != nullptr; }

    template<typename T>
    T GetSymbol(std::string_view symbol) {
        if (!m_handle) return nullptr;
#if defined(_WIN32)
        return reinterpret_cast<T>(GetProcAddress(static_cast<HMODULE>(m_handle), symbol.data()));
#else
        return reinterpret_cast<T>(dlsym(m_handle, symbol.data()));
#endif
    }

private:
    void* m_handle = nullptr;
};

} // namespace DynamicModule