#include "HermeticNetworkingEngine.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>

using namespace Type0;

// ENet transport wrapped with deterministic packet recording and bit-exact state validation
class ENetDeterministicReplayEngine {
public:
    struct RecordedENetFrame {
        uint32_t tick;
        uint32_t sequence;
        uint8_t channel;
        std::vector<uint8_t> payload;
    };

    ENetDeterministicReplayEngine() = default;

    void SetMode(bool recording, bool replaying) {
        m_isRecording = recording;
        m_isReplaying = replaying;
    }

    void Reset() {
        m_replayIndex = 0;
        m_currentTick = 0;
    }

    void SimulateTick(uint32_t tick) {
        m_currentTick = tick;
        m_enetTransport.SimulateTick(tick);
    }

    bool SendState(uint8_t channel, const std::vector<NetEntityState>& entities) {
        size_t size = entities.size() * sizeof(NetEntityState);
        const void* data = entities.data();

        if (m_isReplaying) return false;

        if (m_isRecording) {
            RecordedENetFrame frame;
            frame.tick = m_currentTick;
            frame.sequence = m_seq++;
            frame.channel = channel;
            frame.payload.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
            m_recordedStream.push_back(std::move(frame));
        }

        return m_enetTransport.Send(channel, data, size, PacketFlags::Reliable);
    }

    size_t ReceiveState(std::vector<NetEntityState>& outEntities, uint8_t& outChannel) {
        size_t maxBytes = outEntities.size() * sizeof(NetEntityState);

        if (m_isReplaying) {
            if (m_replayIndex >= m_recordedStream.size()) return 0;
            const auto& frame = m_recordedStream[m_replayIndex];
            if (frame.tick > m_currentTick) return 0;

            outChannel = frame.channel;
            size_t copySize = std::min(maxBytes, frame.payload.size());
            std::memcpy(outEntities.data(), frame.payload.data(), copySize);
            m_replayIndex++;
            return copySize;
        }

        return m_enetTransport.Receive(outEntities.data(), maxBytes, outChannel);
    }

    size_t GetRecordedFrameCount() const { return m_recordedStream.size(); }
    const std::vector<RecordedENetFrame>& GetRecordedStream() const { return m_recordedStream; }

private:
    ENetSimulatedTransport m_enetTransport;
    bool m_isRecording = false;
    bool m_isReplaying = false;
    uint32_t m_currentTick = 0;
    uint32_t m_seq = 0;
    size_t m_replayIndex = 0;
    std::vector<RecordedENetFrame> m_recordedStream;
};

void BenchmarkENetBitExactDeterminism(const std::vector<NetEntityState>& baseEntities, uint32_t tickCount) {
    std::cout << "\n======================================================\n";
    std::cout << " ENet (Simple UDP Channels) Bit-Exact Performance Test\n";
    std::cout << " Workload: " << baseEntities.size() << " Entities across " << tickCount << " Ticks\n";
    std::cout << "======================================================\n";

    ENetDeterministicReplayEngine enetEngine;

    // --- Phase 1: Live Record Session ---
    enetEngine.SetMode(true, false);
    std::vector<uint64_t> liveHashes;
    liveHashes.reserve(tickCount);
    std::vector<NetEntityState> liveEntities = baseEntities;

    size_t totalBytesTransferred = 0;
    auto recordStart = std::chrono::high_resolution_clock::now();

    for (uint32_t tick = 0; tick < tickCount; ++tick) {
        enetEngine.SimulateTick(tick);

        // Update physics positions
        for (auto& e : liveEntities) {
            e.posX += e.velX * 0.016f;
            e.posY += e.velY * 0.016f;
            e.posZ += e.velZ * 0.016f;
            e.lastProcessedTick = tick;
        }

        enetEngine.SendState(0, liveEntities);
        totalBytesTransferred += (liveEntities.size() * sizeof(NetEntityState)) + 12; // 12-byte ENet header

        std::vector<NetEntityState> recvBuf(liveEntities.size());
        uint8_t channel = 0;
        size_t bytes = enetEngine.ReceiveState(recvBuf, channel);
        assert(bytes == liveEntities.size() * sizeof(NetEntityState));

        uint64_t tickHash = HashNetworkState(recvBuf);
        liveHashes.push_back(tickHash);
    }

    auto recordEnd = std::chrono::high_resolution_clock::now();
    double recordDurationMs = std::chrono::duration<double, std::milli>(recordEnd - recordStart).count();

    // --- Phase 2: Bit-Exact Deterministic Replay ---
    enetEngine.SetMode(false, true);
    enetEngine.Reset();

    std::vector<uint64_t> replayHashes;
    replayHashes.reserve(tickCount);

    auto replayStart = std::chrono::high_resolution_clock::now();
    bool bitExactMatch = true;

    for (uint32_t tick = 0; tick < tickCount; ++tick) {
        enetEngine.SimulateTick(tick);

        std::vector<NetEntityState> replayedBuf(baseEntities.size());
        uint8_t channel = 0;
        size_t bytes = enetEngine.ReceiveState(replayedBuf, channel);
        assert(bytes == baseEntities.size() * sizeof(NetEntityState));

        uint64_t replayHash = HashNetworkState(replayedBuf);
        replayHashes.push_back(replayHash);

        if (replayHash != liveHashes[tick]) {
            std::cerr << " [ERROR] ENet state mismatch at tick " << tick << "\n";
            bitExactMatch = false;
            break;
        }
    }

    auto replayEnd = std::chrono::high_resolution_clock::now();
    double replayDurationMs = std::chrono::duration<double, std::milli>(replayEnd - replayStart).count();

    double totalMB = (totalBytesTransferred / (1024.0 * 1024.0));
    double recordMBs = totalMB / (recordDurationMs / 1000.0);
    double replayMBs = totalMB / (replayDurationMs / 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << " [ENet Record] Time: " << recordDurationMs << " ms | Throughput: " << recordMBs << " MB/s\n";
    std::cout << " [ENet Replay] Time: " << replayDurationMs << " ms | Throughput: " << replayMBs << " MB/s\n";
    std::cout << " [Recorded Frames]: " << enetEngine.GetRecordedFrameCount() << " frames\n";

    if (bitExactMatch) {
        std::cout << " [Result]: 100% BIT-EXACT DETERMINISTIC REPLAY PASSED!\n";
        std::cout << " [Initial Tick 0 Hash] : 0x" << std::hex << liveHashes[0] << "\n";
        std::cout << " [Final Tick " << (tickCount - 1) << " Hash]: 0x" << liveHashes.back() << std::dec << "\n";
    } else {
        std::cerr << " [Result]: DETERMINISM FAILURE!\n";
    }
}

int main() {
    std::cout << "======================================================\n";
    std::cout << " ENet Simple UDP Channels - Bit-Exact Determinism Benchmark\n";
    std::cout << "======================================================\n";

    constexpr size_t numEntities = 10000; // Scaled to 10,000 entities
    constexpr uint32_t tickCount = 1000;

    std::vector<NetEntityState> entities(numEntities);
    for (size_t i = 0; i < numEntities; ++i) {
        entities[i].entityId = static_cast<uint32_t>(i + 1);
        entities[i].posX = static_cast<float>(i % 100);
        entities[i].posY = static_cast<float>((i * 3) % 100);
        entities[i].posZ = static_cast<float>((i * 7) % 100);
        entities[i].velX = 2.5f;
        entities[i].velY = -1.2f;
        entities[i].velZ = 0.8f;
        entities[i].lastProcessedTick = 0;
    }

    BenchmarkENetBitExactDeterminism(entities, tickCount);

    return 0;
}
