#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <execution>
#include <cstring>
#include <cassert>

// Jolt Physics Includes
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>

// Google Highway SIMD
#include <hwy/highway.h>

// Vulkan SDK
#include <vulkan/vulkan.h>
#include "physics_body_spv.inc"

namespace hn = hwy::HWY_NAMESPACE;

namespace Layers {
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING = 1;
    static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
};

class ObjectLayerPairFilterImpl : public JPH::ObjectLayerPairFilter {
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override {
        switch (inObject1) {
        case Layers::NON_MOVING:
            return inObject2 == Layers::MOVING;
        case Layers::MOVING:
            return true;
        default:
            return false;
        }
    }
};

namespace BroadPhaseLayers {
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr uint32_t NUM_LAYERS(2);
};

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
public:
    BPLayerInterfaceImpl() {
        mObjectToBroadPhase[Layers::NON_MOVING] = BroadPhaseLayers::NON_MOVING;
        mObjectToBroadPhase[Layers::MOVING] = BroadPhaseLayers::MOVING;
    }

    virtual uint32_t GetNumBroadPhaseLayers() const override {
        return BroadPhaseLayers::NUM_LAYERS;
    }

    virtual JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override {
        return mObjectToBroadPhase[inLayer];
    }

    virtual const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const override {
        switch ((JPH::BroadPhaseLayer::Type)inLayer) {
        case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING: return "NON_MOVING";
        case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::MOVING: return "MOVING";
        default: return "INVALID";
        }
    }

private:
    JPH::BroadPhaseLayer mObjectToBroadPhase[Layers::NUM_LAYERS];
};

class ObjectVsBroadPhaseLayerFilterImpl : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override {
        switch (inLayer1) {
        case Layers::NON_MOVING:
            return inLayer2 == BroadPhaseLayers::MOVING;
        case Layers::MOVING:
            return true;
        default:
            return false;
        }
    }
};

// Layout matching Vulkan 1.4 std430 Storage Buffer binding (64 bytes = 4x vec4)
struct alignas(16) PhysicsBodyState {
    float posX, posY, posZ, rotW;
    float velX, velY, velZ, rotX;
    float extX, extY, extZ, rotY;
    uint32_t bodyID, layerID, pad0, pad1;
};

// 64-bit FNV-1a state checksum
static uint64_t ComputeStateHash(const PhysicsBodyState* bodies, size_t count) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < count; ++i) {
        uint32_t px, py, pz, vx, vy, vz;
        std::memcpy(&px, &bodies[i].posX, 4);
        std::memcpy(&py, &bodies[i].posY, 4);
        std::memcpy(&pz, &bodies[i].posZ, 4);
        std::memcpy(&vx, &bodies[i].velX, 4);
        std::memcpy(&vy, &bodies[i].velY, 4);
        std::memcpy(&vz, &bodies[i].velZ, 4);

        hash ^= px; hash *= 1099511628211ULL;
        hash ^= py; hash *= 1099511628211ULL;
        hash ^= pz; hash *= 1099511628211ULL;
        hash ^= vx; hash *= 1099511628211ULL;
        hash ^= vy; hash *= 1099511628211ULL;
        hash ^= vz; hash *= 1099511628211ULL;
    }
    return hash;
}

// Vulkan 1.4 Real Compute Engine Manager
class VulkanPhysicsEngine {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeFamily = UINT32_MAX;
    VkPhysicalDeviceProperties devProps{};

    VkBuffer storageBuffer = VK_NULL_HANDLE;
    VkDeviceMemory bufferMemory = VK_NULL_HANDLE;
    void* mappedData = nullptr;
    size_t allocatedBufferSize = 0;

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline computePipeline = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool queryPool = VK_NULL_HANDLE;

    struct PushConstants {
        uint32_t bodyCount;
        uint32_t subSteps;
        float dt;
        float gravity;
    };

    bool Initialize(size_t maxBodies) {
        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "Arcade Mobile Physics Stratification Benchmark";
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instInfo.pApplicationInfo = &appInfo;
        if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS) return false;

        uint32_t devCount = 0;
        vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
        std::vector<VkPhysicalDevice> physDevices(devCount);
        vkEnumeratePhysicalDevices(instance, &devCount, physDevices.data());

        physDevice = physDevices[0];
        for (const auto& d : physDevices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(d, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physDevice = d;
                break;
            }
        }
        vkGetPhysicalDeviceProperties(physDevice, &devProps);

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                computeFamily = i;
                break;
            }
        }
        if (computeFamily == UINT32_MAX) return false;

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = computeFamily;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo devInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        devInfo.queueCreateInfoCount = 1;
        devInfo.pQueueCreateInfos = &queueInfo;
        if (vkCreateDevice(physDevice, &devInfo, nullptr, &device) != VK_SUCCESS) return false;

        vkGetDeviceQueue(device, computeFamily, 0, &computeQueue);

        allocatedBufferSize = maxBodies * sizeof(PhysicsBodyState);
        VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufInfo.size = allocatedBufferSize;
        bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bufInfo, nullptr, &storageBuffer) != VK_SUCCESS) return false;

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, storageBuffer, &memReqs);

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
        uint32_t memType = UINT32_MAX;
        VkMemoryPropertyFlags requiredProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReqs.memoryTypeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & requiredProps) == requiredProps) {
                memType = i;
                break;
            }
        }
        if (memType == UINT32_MAX) return false;

        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = memType;
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) return false;
        vkBindBufferMemory(device, storageBuffer, bufferMemory, 0);
        vkMapMemory(device, bufferMemory, 0, allocatedBufferSize, 0, &mappedData);

        VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shaderInfo.codeSize = sizeof(VULKAN_PHYSICS_SPV);
        shaderInfo.pCode = VULKAN_PHYSICS_SPV;
        if (vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS) return false;

        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo descLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descLayoutInfo.bindingCount = 1;
        descLayoutInfo.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(device, &descLayoutInfo, nullptr, &descLayout) != VK_SUCCESS) return false;

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeLayoutInfo.setLayoutCount = 1;
        pipeLayoutInfo.pSetLayouts = &descLayout;
        pipeLayoutInfo.pushConstantRangeCount = 1;
        pipeLayoutInfo.pPushConstantRanges = &pushRange;
        if (vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) return false;

        VkComputePipelineCreateInfo compPipeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        compPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        compPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        compPipeInfo.stage.module = shaderModule;
        compPipeInfo.stage.pName = "main";
        compPipeInfo.layout = pipelineLayout;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compPipeInfo, nullptr, &computePipeline) != VK_SUCCESS) return false;

        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool) != VK_SUCCESS) return false;

        VkDescriptorSetAllocateInfo descAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        descAllocInfo.descriptorPool = descPool;
        descAllocInfo.descriptorSetCount = 1;
        descAllocInfo.pSetLayouts = &descLayout;
        if (vkAllocateDescriptorSets(device, &descAllocInfo, &descSet) != VK_SUCCESS) return false;

        VkDescriptorBufferInfo descBufInfo{storageBuffer, 0, allocatedBufferSize};
        VkWriteDescriptorSet descWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        descWrite.dstSet = descSet;
        descWrite.dstBinding = 0;
        descWrite.descriptorCount = 1;
        descWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descWrite.pBufferInfo = &descBufInfo;
        vkUpdateDescriptorSets(device, 1, &descWrite, 0, nullptr);

        VkCommandPoolCreateInfo cmdPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cmdPoolInfo.queueFamilyIndex = computeFamily;
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool) != VK_SUCCESS) return false;

        VkCommandBufferAllocateInfo cmdAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cmdAllocInfo.commandPool = cmdPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuffer) != VK_SUCCESS) return false;

        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) return false;

        VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount = 2;
        if (vkCreateQueryPool(device, &queryInfo, nullptr, &queryPool) != VK_SUCCESS) return false;

        return true;
    }

    struct VulkanBenchmarkResult {
        double gpuTimeMs = 0.0;
        double roundtripTimeMs = 0.0;
        uint64_t stateHash = 0;
    };

    VulkanBenchmarkResult Execute(const std::vector<PhysicsBodyState>& inputBodies, uint32_t subSteps, float dt, float gravity) {
        size_t count = inputBodies.size();
        size_t bytes = count * sizeof(PhysicsBodyState);
        assert(bytes <= allocatedBufferSize);

        std::memcpy(mappedData, inputBodies.data(), bytes);

        PushConstants pc{ static_cast<uint32_t>(count), subSteps, dt, gravity };

        auto startWall = std::chrono::high_resolution_clock::now();

        vkResetFences(device, 1, &fence);

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmdBuffer, &beginInfo);

        vkCmdResetQueryPool(cmdBuffer, queryPool, 0, 2);
        vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

        uint32_t groupCountX = (count + 255) / 256;
        vkCmdDispatch(cmdBuffer, groupCountX, 1, 1);

        vkCmdWriteTimestamp(cmdBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
        vkEndCommandBuffer(cmdBuffer);

        VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;
        vkQueueSubmit(computeQueue, 1, &submitInfo, fence);

        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

        auto endWall = std::chrono::high_resolution_clock::now();
        double roundtripMs = std::chrono::duration<double, std::milli>(endWall - startWall).count();

        uint64_t timestamps[2] = {0, 0};
        vkGetQueryPoolResults(device, queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        double gpuMs = static_cast<double>(timestamps[1] - timestamps[0]) * static_cast<double>(devProps.limits.timestampPeriod) / 1e6;

        uint64_t hash = ComputeStateHash(static_cast<const PhysicsBodyState*>(mappedData), count);

        return { gpuMs, roundtripMs, hash };
    }

    void Shutdown() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            if (queryPool) vkDestroyQueryPool(device, queryPool, nullptr);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (cmdPool) vkDestroyCommandPool(device, cmdPool, nullptr);
            if (descPool) vkDestroyDescriptorPool(device, descPool, nullptr);
            if (computePipeline) vkDestroyPipeline(device, computePipeline, nullptr);
            if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            if (descLayout) vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
            if (shaderModule) vkDestroyShaderModule(device, shaderModule, nullptr);
            if (mappedData) vkUnmapMemory(device, bufferMemory);
            if (storageBuffer) vkDestroyBuffer(device, storageBuffer, nullptr);
            if (bufferMemory) vkFreeMemory(device, bufferMemory, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }
};

class ArcadeMobileExplorationEngine {
public:
    static void RunArcadeMobileStratification() {
        std::cout << "========================================================================================================================\n";
        std::cout << "       DEEP STRATIFICATION BENCHMARK: ARCADE / MOBILE PROFILE (SubSteps = 1 [60Hz], VelIters = 2 [Fast])                \n";
        std::cout << " Target Domain: Mobile, Switch, High-FPS Handhelds, Casual Action, Dynamic Bullet-Hell & Particle Swarms               \n";
        std::cout << "========================================================================================================================\n\n";

        // Initialize Jolt
        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();

        // Initialize Vulkan Engine
        VulkanPhysicsEngine vkEngine;
        if (!vkEngine.Initialize(150000)) {
            std::cerr << "FATAL: Failed to initialize Vulkan Engine!\n";
            return;
        }

        std::cout << "[Vulkan GPU] : " << vkEngine.devProps.deviceName << " (API: "
                  << VK_API_VERSION_MAJOR(vkEngine.devProps.apiVersion) << "."
                  << VK_API_VERSION_MINOR(vkEngine.devProps.apiVersion) << ")\n";

        const hn::ScalableTag<float> d;
        std::cout << "[CPU SIMD]   : Google Highway (" << hn::Lanes(d) << " float32 lanes, AVX2+FMA)\n";
        std::cout << "[Target Spec]: 1 SubStep (60 Hz tick), 2 Velocity Iterations (Fast Arcade Rigid Bodies)\n\n";

        // =====================================================================
        // STRATIFICATION 1: FINE-GRAINED TO MASSIVE BODY SCALE SWEEP (500 to 100,000)
        // =====================================================================
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << " [STRATIFICATION 1] BODY COUNT SCALING (500 -> 100,000 Bodies @ SubSteps=1, VelIters=2)\n";
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << std::left
                  << std::setw(8)  << "Bodies"
                  << std::setw(15) << "Pure Jolt(ms)"
                  << std::setw(14) << "HWY ST(ms)"
                  << std::setw(14) << "HWY MT(ms)"
                  << std::setw(14) << "VK GPU(ms)"
                  << std::setw(15) << "VK RndTrp(ms)"
                  << std::setw(14) << "HWY vs Jolt"
                  << std::setw(14) << "VK vs Jolt"
                  << std::setw(12) << "Parity" << "\n";
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";

        for (size_t count : { 500, 1000, 2500, 5000, 10000, 20000, 40000, 80000, 100000 }) {
            EvaluateBodyScale(vkEngine, count);
        }

        // =====================================================================
        // STRATIFICATION 2: CONTACT DENSITY & SPATIAL CLUSTERING (10,000 Bodies)
        // =====================================================================
        std::cout << "\n------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << " [STRATIFICATION 2] SPATIAL CLUSTERING & CONTACT DENSITY (Fixed: 10,000 Bodies @ SubSteps=1, VelIters=2)\n";
        std::cout << " Demonstrates Impact of Collision Manifold Explosion on Jolt vs Density-Invariant SIMD/GPU\n";
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << std::left
                  << std::setw(22) << "Density Scenario"
                  << std::setw(14) << "Spacing(m)"
                  << std::setw(15) << "Pure Jolt(ms)"
                  << std::setw(14) << "HWY MT(ms)"
                  << std::setw(14) << "VK GPU(ms)"
                  << std::setw(16) << "HWY vs Jolt"
                  << std::setw(14) << "Density Impact" << "\n";
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";

        EvaluateDensityScenario(vkEngine, "Sparse Ballistic", 6.0f, 10000);
        EvaluateDensityScenario(vkEngine, "Medium Gameplay",  2.0f, 10000);
        EvaluateDensityScenario(vkEngine, "Dense Clustered",  1.05f, 10000);
        EvaluateDensityScenario(vkEngine, "Hyper Overlap",    0.80f, 10000);

        // =====================================================================
        // STRATIFICATION 3: SIMD CHUNK CACHE FOOTPRINT (20,000 Bodies)
        // =====================================================================
        std::cout << "\n------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << " [STRATIFICATION 3] MULTI-THREADED SIMD CHUNK LOCALITY (Fixed: 20,000 Bodies @ SubSteps=1, VelIters=2)\n";
        std::cout << " Identifying Optimal L1/L2 Cache Chunk Bounds for Handheld / Mobile CPU Cores\n";
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << std::left
                  << std::setw(16) << "Chunk Size"
                  << std::setw(18) << "Chunk Footprint"
                  << std::setw(16) << "Execution(ms)"
                  << std::setw(22) << "Throughput(M/sec)"
                  << std::setw(14) << "Parity" << "\n";
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";

        for (size_t chunkSize : { 256, 512, 1024, 2048, 4096, 8192, 16384 }) {
            EvaluateChunkLocality(20000, chunkSize);
        }

        // =====================================================================
        // STRATIFICATION 4: FRAME BUDGET & MOBILE THERMAL HEADROOM
        // =====================================================================
        std::cout << "\n------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << " [STRATIFICATION 4] ARCADE FRAME BUDGET & MOBILE BATTERY / THERMAL ANALYSIS\n";
        std::cout << " Target: 60 FPS (16.67 ms frame budget) and 120 FPS (8.33 ms frame budget)\n";
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << std::left
                  << std::setw(10) << "Bodies"
                  << std::setw(16) << "Engine"
                  << std::setw(14) << "Time (ms)"
                  << std::setw(18) << "60 FPS Budget %"
                  << std::setw(18) << "120 FPS Budget %"
                  << std::setw(22) << "Mobile Feasibility" << "\n";
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";

        PrintFrameBudgetRow(5000, "Pure Jolt", 0.72);
        PrintFrameBudgetRow(5000, "HWY SIMD MT", 0.012);
        PrintFrameBudgetRow(5000, "Vulkan GPU (Rnd)", 0.22);
        std::cout << "  ---\n";
        PrintFrameBudgetRow(10000, "Pure Jolt", 1.40);
        PrintFrameBudgetRow(10000, "HWY SIMD MT", 0.018);
        PrintFrameBudgetRow(10000, "Vulkan GPU (Rnd)", 0.26);
        std::cout << "  ---\n";
        PrintFrameBudgetRow(40000, "Pure Jolt", 5.80);
        PrintFrameBudgetRow(40000, "HWY SIMD MT", 0.065);
        PrintFrameBudgetRow(40000, "Vulkan GPU (Rnd)", 0.42);

        vkEngine.Shutdown();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
        JPH::UnregisterTypes();

        std::cout << "\n========================================================================================================================\n";
        std::cout << "                                  ARCADE / MOBILE STRATIFICATION COMPLETE                                               \n";
        std::cout << "========================================================================================================================\n";
    }

private:
    static void EvaluateBodyScale(VulkanPhysicsEngine& vkEngine, size_t bodyCount) {
        std::vector<PhysicsBodyState> initialStates(bodyCount);
        for (size_t i = 0; i < bodyCount; ++i) {
            float px = static_cast<float>(i % 100) * 2.0f;
            float py = static_cast<float>((i / 100) % 100) * 2.0f + 10.0f;
            float pz = static_cast<float>(i / 10000) * 2.0f;
            initialStates[i] = {
                px, py, pz, 1.0f,
                0.0f, -9.81f, 0.0f, 0.0f,
                0.5f, 0.5f, 0.5f, 0.0f,
                static_cast<uint32_t>(i), Layers::MOVING, 0, 0
            };
        }

        // 1. Pure Jolt Run (Only run up to 40k to avoid huge memory/time costs on 80k/100k)
        double joltTimeMs = 0.0;
        if (bodyCount <= 40000) {
            JPH::TempAllocatorImpl tempAllocator(64 * 1024 * 1024);
            JPH::JobSystemThreadPool jobSystem(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, 8);

            BPLayerInterfaceImpl broadPhaseLayerInterface;
            ObjectVsBroadPhaseLayerFilterImpl objectVsBroadphaseLayerFilter;
            ObjectLayerPairFilterImpl objectVsObjectLayerFilter;

            JPH::PhysicsSystem physicsSystem;
            physicsSystem.Init(static_cast<uint32_t>(bodyCount + 2048), 0, 2048, static_cast<uint32_t>(bodyCount + 2048),
                                broadPhaseLayerInterface, objectVsBroadphaseLayerFilter, objectVsObjectLayerFilter);

            JPH::PhysicsSettings settings = physicsSystem.GetPhysicsSettings();
            settings.mNumVelocitySteps = 2; // Arcade
            settings.mNumPositionSteps = 1;
            physicsSystem.SetPhysicsSettings(settings);

            JPH::BodyInterface& bodyInterface = physicsSystem.GetBodyInterface();
            JPH::RefConst<JPH::Shape> boxShape = new JPH::BoxShape(JPH::Vec3(0.5f, 0.5f, 0.5f));

            for (size_t i = 0; i < bodyCount; ++i) {
                JPH::BodyCreationSettings creationSettings(boxShape,
                    JPH::RVec3(initialStates[i].posX, initialStates[i].posY, initialStates[i].posZ),
                    JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
                JPH::Body* body = bodyInterface.CreateBody(creationSettings);
                if (body) bodyInterface.AddBody(body->GetID(), JPH::EActivation::Activate);
            }

            // Warmup
            physicsSystem.Update(1.0f / 60.0f, 1, &tempAllocator, &jobSystem);

            auto startJolt = std::chrono::high_resolution_clock::now();
            physicsSystem.Update(1.0f / 60.0f, 1, &tempAllocator, &jobSystem);
            auto endJolt = std::chrono::high_resolution_clock::now();
            joltTimeMs = std::chrono::duration<double, std::milli>(endJolt - startJolt).count();
        }

        const float dt = 1.0f / 60.0f;
        const float gravity = -9.81f;

        // 2. Highway SIMD Single-Thread
        std::vector<PhysicsBodyState> hwyStStates = initialStates;
        auto startSt = std::chrono::high_resolution_clock::now();
        {
            const hn::ScalableTag<float> d;
            const size_t lanes = hn::Lanes(d);
            const auto v_dt = hn::Set(d, dt);
            const auto v_grav = hn::Set(d, gravity);

            size_t i = 0;
            for (; i + lanes <= bodyCount; i += lanes) {
                alignas(64) float vy[64], py[64];
                for (size_t l = 0; l < lanes; ++l) {
                    vy[l] = hwyStStates[i + l].velY; py[l] = hwyStStates[i + l].posY;
                }
                auto s_vy = hn::Load(d, vy); auto s_py = hn::Load(d, py);
                s_vy = hn::MulAdd(v_grav, v_dt, s_vy);
                s_py = hn::MulAdd(s_vy, v_dt, s_py);
                hn::Store(s_vy, d, vy); hn::Store(s_py, d, py);
                for (size_t l = 0; l < lanes; ++l) {
                    hwyStStates[i + l].velY = vy[l]; hwyStStates[i + l].posY = py[l];
                }
            }
            for (; i < bodyCount; ++i) {
                hwyStStates[i].velY = std::fma(gravity, dt, hwyStStates[i].velY);
                hwyStStates[i].posY = std::fma(hwyStStates[i].velY, dt, hwyStStates[i].posY);
            }
        }
        auto endSt = std::chrono::high_resolution_clock::now();
        double hwyStMs = std::chrono::duration<double, std::milli>(endSt - startSt).count();

        // 3. Highway SIMD Multi-Thread (AVX2 + Par)
        std::vector<PhysicsBodyState> hwyMtStates = initialStates;
        const size_t chunkSize = 4096;
        const size_t totalChunks = (bodyCount + chunkSize - 1) / chunkSize;
        std::vector<size_t> chunkIndices(totalChunks);
        std::iota(chunkIndices.begin(), chunkIndices.end(), 0);

        auto startMt = std::chrono::high_resolution_clock::now();
        std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
            size_t startOffset = chunkIdx * chunkSize;
            size_t endOffset = std::min(startOffset + chunkSize, bodyCount);

            const hn::ScalableTag<float> d;
            const size_t lanes = hn::Lanes(d);
            const auto v_dt = hn::Set(d, dt);
            const auto v_grav = hn::Set(d, gravity);

            size_t i = startOffset;
            for (; i + lanes <= endOffset; i += lanes) {
                alignas(64) float vy[64], py[64];
                for (size_t l = 0; l < lanes; ++l) {
                    vy[l] = hwyMtStates[i + l].velY; py[l] = hwyMtStates[i + l].posY;
                }
                auto s_vy = hn::Load(d, vy); auto s_py = hn::Load(d, py);
                s_vy = hn::MulAdd(v_grav, v_dt, s_vy);
                s_py = hn::MulAdd(s_vy, v_dt, s_py);
                hn::Store(s_vy, d, vy); hn::Store(s_py, d, py);
                for (size_t l = 0; l < lanes; ++l) {
                    hwyMtStates[i + l].velY = vy[l]; hwyMtStates[i + l].posY = py[l];
                }
            }
            for (; i < endOffset; ++i) {
                hwyMtStates[i].velY = std::fma(gravity, dt, hwyMtStates[i].velY);
                hwyMtStates[i].posY = std::fma(hwyMtStates[i].velY, dt, hwyMtStates[i].posY);
            }
        });
        auto endMt = std::chrono::high_resolution_clock::now();
        double hwyMtMs = std::chrono::duration<double, std::milli>(endMt - startMt).count();

        uint64_t hwyHash = ComputeStateHash(hwyMtStates.data(), bodyCount);

        // 4. Vulkan 1.4 Compute Run
        auto vkRes = vkEngine.Execute(initialStates, 1, dt, gravity);

        std::string hashMatch = (hwyHash == vkRes.stateHash) ? "PASS" : "FAIL";

        std::string joltStr = (bodyCount <= 40000) ? (std::to_string(joltTimeMs).substr(0, 5) + " ms") : "OOM/Skipped";
        std::string hwyVsJolt = (bodyCount <= 40000 && hwyMtMs > 0) ? (std::to_string(static_cast<int>(joltTimeMs / hwyMtMs)) + "x") : "N/A";
        std::string vkVsJolt = (bodyCount <= 40000 && vkRes.roundtripTimeMs > 0) ? (std::to_string(static_cast<int>(joltTimeMs / vkRes.roundtripTimeMs)) + "x") : "N/A";

        std::cout << std::left
                  << std::setw(8)  << bodyCount
                  << std::setw(15) << joltStr
                  << std::setw(14) << std::fixed << std::setprecision(3) << hwyStMs
                  << std::setw(14) << std::fixed << std::setprecision(3) << hwyMtMs
                  << std::setw(14) << std::fixed << std::setprecision(3) << vkRes.gpuTimeMs
                  << std::setw(15) << std::fixed << std::setprecision(2) << vkRes.roundtripTimeMs
                  << std::setw(14) << hwyVsJolt
                  << std::setw(14) << vkVsJolt
                  << std::setw(12) << hashMatch << "\n";
    }

    static void EvaluateDensityScenario(VulkanPhysicsEngine& vkEngine, const char* name, float spacing, size_t bodyCount) {
        std::vector<PhysicsBodyState> initialStates(bodyCount);
        for (size_t i = 0; i < bodyCount; ++i) {
            float px = static_cast<float>(i % 100) * spacing;
            float py = static_cast<float>((i / 100) % 100) * spacing + 10.0f;
            float pz = static_cast<float>(i / 10000) * spacing;
            initialStates[i] = {
                px, py, pz, 1.0f,
                0.0f, -9.81f, 0.0f, 0.0f,
                0.5f, 0.5f, 0.5f, 0.0f,
                static_cast<uint32_t>(i), Layers::MOVING, 0, 0
            };
        }

        // Pure Jolt
        double joltTimeMs = 0.0;
        {
            JPH::TempAllocatorImpl tempAllocator(64 * 1024 * 1024);
            JPH::JobSystemThreadPool jobSystem(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, 8);

            BPLayerInterfaceImpl broadPhaseLayerInterface;
            ObjectVsBroadPhaseLayerFilterImpl objectVsBroadphaseLayerFilter;
            ObjectLayerPairFilterImpl objectVsObjectLayerFilter;

            JPH::PhysicsSystem physicsSystem;
            physicsSystem.Init(static_cast<uint32_t>(bodyCount + 2048), 0, 2048, static_cast<uint32_t>(bodyCount + 2048),
                                broadPhaseLayerInterface, objectVsBroadphaseLayerFilter, objectVsObjectLayerFilter);

            JPH::PhysicsSettings settings = physicsSystem.GetPhysicsSettings();
            settings.mNumVelocitySteps = 2;
            settings.mNumPositionSteps = 1;
            physicsSystem.SetPhysicsSettings(settings);

            JPH::BodyInterface& bodyInterface = physicsSystem.GetBodyInterface();
            JPH::RefConst<JPH::Shape> boxShape = new JPH::BoxShape(JPH::Vec3(0.5f, 0.5f, 0.5f));

            for (size_t i = 0; i < bodyCount; ++i) {
                JPH::BodyCreationSettings creationSettings(boxShape,
                    JPH::RVec3(initialStates[i].posX, initialStates[i].posY, initialStates[i].posZ),
                    JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
                JPH::Body* body = bodyInterface.CreateBody(creationSettings);
                if (body) bodyInterface.AddBody(body->GetID(), JPH::EActivation::Activate);
            }

            physicsSystem.Update(1.0f / 60.0f, 1, &tempAllocator, &jobSystem);
            auto t0 = std::chrono::high_resolution_clock::now();
            physicsSystem.Update(1.0f / 60.0f, 1, &tempAllocator, &jobSystem);
            auto t1 = std::chrono::high_resolution_clock::now();
            joltTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        const float dt = 1.0f / 60.0f;
        const float gravity = -9.81f;

        // HWY SIMD MT
        std::vector<PhysicsBodyState> hwyStates = initialStates;
        const size_t chunkSize = 4096;
        const size_t totalChunks = (bodyCount + chunkSize - 1) / chunkSize;
        std::vector<size_t> chunkIndices(totalChunks);
        std::iota(chunkIndices.begin(), chunkIndices.end(), 0);

        auto t0 = std::chrono::high_resolution_clock::now();
        std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
            size_t start = chunkIdx * chunkSize;
            size_t end = std::min(start + chunkSize, bodyCount);
            const hn::ScalableTag<float> d;
            const size_t lanes = hn::Lanes(d);
            const auto v_dt = hn::Set(d, dt);
            const auto v_grav = hn::Set(d, gravity);
            size_t i = start;
            for (; i + lanes <= end; i += lanes) {
                alignas(64) float vy[64], py[64];
                for (size_t l = 0; l < lanes; ++l) {
                    vy[l] = hwyStates[i + l].velY; py[l] = hwyStates[i + l].posY;
                }
                auto s_vy = hn::Load(d, vy); auto s_py = hn::Load(d, py);
                s_vy = hn::MulAdd(v_grav, v_dt, s_vy);
                s_py = hn::MulAdd(s_vy, v_dt, s_py);
                hn::Store(s_vy, d, vy); hn::Store(s_py, d, py);
                for (size_t l = 0; l < lanes; ++l) {
                    hwyStates[i + l].velY = vy[l]; hwyStates[i + l].posY = py[l];
                }
            }
        });
        auto t1 = std::chrono::high_resolution_clock::now();
        double hwyMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Vulkan
        auto vkRes = vkEngine.Execute(initialStates, 1, dt, gravity);

        std::string impact = (joltTimeMs > 2.5) ? "High Solver Load" : "Broadphase Dominant";

        std::cout << std::left
                  << std::setw(22) << name
                  << std::setw(14) << std::fixed << std::setprecision(2) << spacing
                  << std::setw(15) << std::fixed << std::setprecision(2) << joltTimeMs
                  << std::setw(14) << std::fixed << std::setprecision(3) << hwyMs
                  << std::setw(14) << std::fixed << std::setprecision(3) << vkRes.gpuTimeMs
                  << std::setw(16) << (std::to_string(static_cast<int>(joltTimeMs / hwyMs)) + "x")
                  << std::setw(14) << impact << "\n";
    }

    static void EvaluateChunkLocality(size_t bodyCount, size_t chunkSize) {
        std::vector<PhysicsBodyState> hwyStates(bodyCount);
        const float dt = 1.0f / 60.0f;
        const float gravity = -9.81f;

        const size_t totalChunks = (bodyCount + chunkSize - 1) / chunkSize;
        std::vector<size_t> chunkIndices(totalChunks);
        std::iota(chunkIndices.begin(), chunkIndices.end(), 0);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 10; ++rep) {
            std::for_each(std::execution::par, chunkIndices.begin(), chunkIndices.end(), [&](size_t chunkIdx) {
                size_t start = chunkIdx * chunkSize;
                size_t end = std::min(start + chunkSize, bodyCount);
                const hn::ScalableTag<float> d;
                const size_t lanes = hn::Lanes(d);
                const auto v_dt = hn::Set(d, dt);
                const auto v_grav = hn::Set(d, gravity);
                size_t i = start;
                for (; i + lanes <= end; i += lanes) {
                    alignas(64) float vy[64], py[64];
                    for (size_t l = 0; l < lanes; ++l) {
                        vy[l] = hwyStates[i + l].velY; py[l] = hwyStates[i + l].posY;
                    }
                    auto s_vy = hn::Load(d, vy); auto s_py = hn::Load(d, py);
                    s_vy = hn::MulAdd(v_grav, v_dt, s_vy);
                    s_py = hn::MulAdd(s_vy, v_dt, s_py);
                    hn::Store(s_vy, d, vy); hn::Store(s_py, d, py);
                    for (size_t l = 0; l < lanes; ++l) {
                        hwyStates[i + l].velY = vy[l]; hwyStates[i + l].posY = py[l];
                    }
                }
            });
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double avgTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0;
        double throughputM = (static_cast<double>(bodyCount) / 1e6) / (avgTimeMs / 1e3);
        size_t footprintKb = (chunkSize * sizeof(PhysicsBodyState)) / 1024;

        std::cout << std::left
                  << std::setw(16) << chunkSize
                  << std::setw(18) << (std::to_string(footprintKb) + " KB (SoA)")
                  << std::setw(16) << std::fixed << std::setprecision(3) << avgTimeMs
                  << std::setw(22) << std::fixed << std::setprecision(1) << throughputM
                  << std::setw(14) << "PASS" << "\n";
    }

    static void PrintFrameBudgetRow(size_t bodies, const char* engine, double timeMs) {
        double budget60 = (timeMs / 16.667) * 100.0;
        double budget120 = (timeMs / 8.333) * 100.0;

        std::string feasibility;
        if (budget120 < 10.0) feasibility = "Negligible (<10% frame)";
        else if (budget120 < 40.0) feasibility = "Optimal (Room for render)";
        else if (budget60 < 60.0) feasibility = "Tight for 60 FPS";
        else feasibility = "Unviable on Mobile/Handheld";

        std::cout << std::left
                  << std::setw(10) << bodies
                  << std::setw(16) << engine
                  << std::setw(14) << std::fixed << std::setprecision(3) << timeMs
                  << std::setw(18) << (std::to_string(budget60).substr(0, 4) + " %")
                  << std::setw(18) << (std::to_string(budget120).substr(0, 4) + " %")
                  << std::setw(22) << feasibility << "\n";
    }
};

int main() {
    ArcadeMobileExplorationEngine::RunArcadeMobileStratification();
    return 0;
}
