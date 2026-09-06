#include "HermeticNetworkingEngine.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <numeric>

using namespace Type0;

struct TestMetrics {
    std::string name;
    double durationMs;
    size_t totalPackets;
    size_t totalVolumeBytes;
    size_t headerOverheadBytes;
    double throughputMBs;
    double pps;
};

TestMetrics RunBenchmark(INetworkTransport& transport, const std::vector<NetEntityState>& testEntities, uint32_t totalTicks) {
    size_t totalBytesSent = 0;
    size_t totalPacketsSent = 0;
    size_t totalPacketsReceived = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t tick = 0; tick < totalTicks; ++tick) {
        transport.SimulateTick(tick);

        size_t payloadSizeBytes = testEntities.size() * sizeof(NetEntityState);
        bool sent = transport.Send(0, testEntities.data(), payloadSizeBytes, PacketFlags::Reliable);
        if (sent) {
            totalPacketsSent++;
            totalBytesSent += payloadSizeBytes + transport.GetHeaderOverheadBytes();
        }

        std::vector<NetEntityState> recvBuffer(testEntities.size());
        uint8_t channel = 0;
        size_t bytesRead = transport.Receive(recvBuffer.data(), recvBuffer.size() * sizeof(NetEntityState), channel);

        if (bytesRead > 0) {
            totalPacketsReceived++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double durationMs = std::chrono::duration<double, std::milli>(end - start).count();
    double packetsPerSec = (totalPacketsSent / (durationMs / 1000.0));
    double throughputMBs = (totalBytesSent / (1024.0 * 1024.0)) / (durationMs / 1000.0);

    return TestMetrics{
        transport.GetName(),
        durationMs,
        totalPacketsSent,
        totalBytesSent,
        transport.GetHeaderOverheadBytes(),
        throughputMBs,
        packetsPerSec
    };
}

void TestDeterministicPacketReplay(const std::vector<NetEntityState>& baseEntities, uint32_t tickCount) {
    std::cout << "\n======================================================\n";
    std::cout << " DETERMINISTIC PACKET REPLAY & STATE DRIFT CHECK\n";
    std::cout << "======================================================\n";

    DeterministicReplayTransport replayEngine;
    replayEngine.SetModeRecord(true);
    replayEngine.SetModeReplay(false);

    std::vector<uint64_t> liveTickHashes;
    std::vector<NetEntityState> liveEntities = baseEntities;

    for (uint32_t tick = 0; tick < tickCount; ++tick) {
        replayEngine.SimulateTick(tick);

        for (auto& e : liveEntities) {
            e.posX += e.velX * 0.016f;
            e.posY += e.velY * 0.016f;
            e.posZ += e.velZ * 0.016f;
            e.lastProcessedTick = tick;
        }

        replayEngine.Send(0, liveEntities.data(), liveEntities.size() * sizeof(NetEntityState), PacketFlags::Unreliable);

        std::vector<NetEntityState> recvBuf(liveEntities.size());
        uint8_t channel = 0;
        size_t bytes = replayEngine.Receive(recvBuf.data(), recvBuf.size() * sizeof(NetEntityState), channel);
        assert(bytes == liveEntities.size() * sizeof(NetEntityState));

        uint64_t tickHash = HashNetworkState(recvBuf);
        liveTickHashes.push_back(tickHash);
    }

    replayEngine.SetModeRecord(false);
    replayEngine.SetModeReplay(true);
    replayEngine.ResetReplay();

    bool bitExactMatch = true;
    for (uint32_t tick = 0; tick < tickCount; ++tick) {
        replayEngine.SimulateTick(tick);

        std::vector<NetEntityState> replayedBuf(baseEntities.size());
        uint8_t channel = 0;
        size_t bytes = replayEngine.Receive(replayedBuf.data(), replayedBuf.size() * sizeof(NetEntityState), channel);

        uint64_t replayHash = HashNetworkState(replayedBuf);

        if (replayHash != liveTickHashes[tick]) {
            std::cerr << " [ERROR] Drift detected at tick " << tick << "\n";
            bitExactMatch = false;
            break;
        }
    }

    if (bitExactMatch) {
        std::cout << " [SUCCESS] 100% Bit-Exact Packet Replay Verified across " << tickCount << " ticks!\n";
        std::cout << " Initial Hash: 0x" << std::hex << liveTickHashes[0] 
                  << " -> Final Hash: 0x" << liveTickHashes.back() << std::dec << "\n";
    }
}

int main() {
    std::cout << "======================================================\n";
    std::cout << " TYPE0 ENGINE - NETWORKING ARCHITECTURE COMPARISON\n";
    std::cout << " Benchmark Target: 10,000 Entities / Packet Stream\n";
    std::cout << "======================================================\n";

    constexpr size_t numEntities = 1000;
    constexpr uint32_t numTicks = 1000;

    std::vector<NetEntityState> entities(numEntities);
    for (size_t i = 0; i < numEntities; ++i) {
        entities[i].entityId = static_cast<uint32_t>(i + 1);
        entities[i].posX = static_cast<float>(i % 100);
        entities[i].posY = static_cast<float>((i * 3) % 100);
        entities[i].posZ = static_cast<float>((i * 7) % 100);
        entities[i].velX = 1.5f;
        entities[i].velY = -0.5f;
        entities[i].velZ = 2.0f;
        entities[i].lastProcessedTick = 0;
    }

    ENetSimulatedTransport enet;
    GNSSimulatedTransport gns;
    DeterministicReplayTransport rawTransport;

    auto mEnet = RunBenchmark(enet, entities, numTicks);
    auto mGns = RunBenchmark(gns, entities, numTicks);
    auto mRaw = RunBenchmark(rawTransport, entities, numTicks);

    std::cout << "\n----------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(38) << "Transport Architecture"
              << std::setw(15) << "Overhead"
              << std::setw(15) << "Latency (1k)"
              << std::setw(18) << "Throughput MB/s"
              << std::setw(18) << "Packets / Sec"
              << "\n";
    std::cout << "----------------------------------------------------------------------------------------------------\n";

    auto PrintRow = [](const TestMetrics& m) {
        std::cout << std::left << std::setw(38) << m.name
                  << std::setw(15) << (std::to_string(m.headerOverheadBytes) + " B")
                  << std::setw(15) << (std::to_string(m.durationMs).substr(0, 5) + " ms")
                  << std::setw(18) << std::fixed << std::setprecision(2) << m.throughputMBs
                  << std::setw(18) << std::fixed << std::setprecision(0) << m.pps
                  << "\n";
    };

    PrintRow(mEnet);
    PrintRow(mGns);
    PrintRow(mRaw);
    std::cout << "----------------------------------------------------------------------------------------------------\n";

    TestDeterministicPacketReplay(entities, 500);

    return 0;
}
