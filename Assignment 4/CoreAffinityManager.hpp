#pragma once
#include <vector>
#include <thread>
#include <iostream>
#include <fstream>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace Type0 {

class CoreAffinityManager {
public:
    static bool PinCurrentThreadToEfficiencyCores() {
#if defined(_WIN32)
        DWORD length = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
        if (length == 0) return false;

        std::vector<uint8_t> buffer(length);
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
        if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
            return false;
        }

        GROUP_AFFINITY affinity{};
        bool foundEcore = false;

        uint8_t* ptr = buffer.data();
        uint8_t* end = ptr + length;
        while (ptr < end) {
            auto* item = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);
            if (item->Relationship == RelationProcessorCore) {
                // Windows marks EfficiencyClass = 0 for efficiency cores (E-cores)
                if (item->Processor.EfficiencyClass == 0) {
                    affinity.Mask |= item->Processor.GroupMask[0].Mask;
                    affinity.Group = item->Processor.GroupMask[0].Group;
                    foundEcore = true;
                }
            }
            ptr += item->Size;
        }

        if (foundEcore) {
            if (SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr)) {
                return true;
            }
        }
        return false;

#else // Linux
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        
        int numCores = sysconf(_SC_NPROCESSORS_ONLN);
        unsigned long long minMaxFreq = ~0ULL;
        std::vector<unsigned long long> coreMaxFreqs(numCores, 0);

        for (int i = 0; i < numCores; ++i) {
            std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/cpuinfo_max_freq";
            std::ifstream file(path);
            unsigned long long freq = 0;
            if (file >> freq) {
                coreMaxFreqs[i] = freq;
                if (freq < minMaxFreq) {
                    minMaxFreq = freq;
                }
            }
        }

        bool pinnedAny = false;
        for (int i = 0; i < numCores; ++i) {
            if (coreMaxFreqs[i] == minMaxFreq && minMaxFreq != ~0ULL) {
                CPU_SET(i, &cpuset);
                pinnedAny = true;
            }
        }

        if (!pinnedAny) {
            for (int i = numCores / 2; i < numCores; ++i) {
                CPU_SET(i, &cpuset);
            }
        }

        return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
#endif
    }
};

} // namespace Type0
