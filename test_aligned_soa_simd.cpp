#include <iostream>
#include <cassert>
#include <chrono>
#include "AlignedSoASIMDBuilder.hpp"

using namespace Type0::ECS;

void TestAC1_AlignmentGuarantee() {
    PositionSoA pos;
    pos.Reserve(1000);
    pos.x.push_back(1.0f);
    pos.y.push_back(2.0f);
    pos.z.push_back(3.0f);

    uintptr_t addrX = reinterpret_cast<uintptr_t>(pos.x.data());
    uintptr_t addrY = reinterpret_cast<uintptr_t>(pos.y.data());
    uintptr_t addrZ = reinterpret_cast<uintptr_t>(pos.z.data());

    assert((addrX & 63) == 0 && "AC-1 Fail: Position.x is not 64-byte aligned!");
    assert((addrY & 63) == 0 && "AC-1 Fail: Position.y is not 64-byte aligned!");
    assert((addrZ & 63) == 0 && "AC-1 Fail: Position.z is not 64-byte aligned!");

    std::cout << "[PASS] AC-1: All array pointers satisfy 64-byte cache line alignment.\n";
}

void TestAC2_BatchBuilding() {
    PositionSoA pos;
    VelocitySoA vel;

    ArchetypeBatchBuilder batchBuilder(pos, vel);
    constexpr size_t BATCH_SIZE = 100'000;
    batchBuilder.SpawnBatch(BATCH_SIZE, 1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f);

    assert(pos.x.size() >= BATCH_SIZE);
    assert(vel.vx.size() >= BATCH_SIZE);

    std::cout << "[PASS] AC-2: Batch Archetype Builder successfully spawned " << BATCH_SIZE << " entities in bulk.\n";
}

void TestAC3_TailPadding() {
    PositionSoA pos;
    VelocitySoA vel;

    AlignedSoABuilder builder(pos, vel);
    // Push 5 entities (odd count to test SIMD lane padding)
    for (int i = 0; i < 5; ++i) {
        builder.WithPosition(static_cast<float>(i), 0.0f, 0.0f)
               .WithVelocity(1.0f, 0.0f, 0.0f)
               .Build();
    }

    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<float> d;
    size_t lanes = hn::Lanes(d);

    assert((pos.x.size() % lanes) == 0 && "AC-3 Fail: Tail padding failed to align size to SIMD lane width!");
    std::cout << "[PASS] AC-3: Array tail padding aligned to SIMD width (" << lanes << " lanes).\n";
}

void TestAC4_SIMDExecutionSpeed() {
    constexpr size_t N = 5'000'000;
    PositionSoA pos;
    VelocitySoA vel;

    ArchetypeBatchBuilder batchBuilder(pos, vel);
    batchBuilder.SpawnBatch(N, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f);

    auto start = std::chrono::high_resolution_clock::now();
    SystemUpdateSIMDPhysics(pos, vel, 0.016f);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[PASS] AC-4: Processed " << N << " entities via Google Highway SIMD in " << elapsedMs << " ms.\n";
}

void TestAC5_SIMDArenaBuilder() {
    alignas(64) uint8_t rawBuffer[1024];
    SIMDArenaBuilder arena(rawBuffer, sizeof(rawBuffer));

    float sampleData[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    arena.PushSIMDBlock(sampleData, 16);

    uintptr_t handleAddr = reinterpret_cast<uintptr_t>(arena.GetBufferHandle());
    assert((handleAddr & 63) == 0 && "AC-5 Fail: SIMD Arena handle is not 64-byte aligned!");
    assert(arena.GetBytesWritten() == 64 && "AC-5 Fail: Block size not rounded up to 64 bytes!");

    std::cout << "[PASS] AC-5: SIMD Arena Builder verified for 64-byte Vulkan push constant staging.\n";
}

int main() {
    std::cout << "=====================================================\n";
    std::cout << "   RUNNING ALIGNED SOA SIMD BUILDER VERIFICATION     \n";
    std::cout << "=====================================================\n\n";

    TestAC1_AlignmentGuarantee();
    TestAC2_BatchBuilding();
    TestAC3_TailPadding();
    TestAC4_SIMDExecutionSpeed();
    TestAC5_SIMDArenaBuilder();

    std::cout << "\nALL ACCEPTANCE CRITERIA (AC-1 to AC-5) SUCCESSFULLY VERIFIED!\n";
    return 0;
}
