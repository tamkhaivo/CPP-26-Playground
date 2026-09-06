#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include <queue>
#include <random>

namespace Type0 {

// Packet flags matching ENet / GameNetworkingSockets features
enum class PacketFlags : uint8_t {
    Unreliable = 0,
    Reliable = 1 << 0,
    Unsequenced = 1 << 1,
    Encrypted = 1 << 2
};

inline PacketFlags operator|(PacketFlags a, PacketFlags b) {
    return static_cast<PacketFlags>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

inline bool operator&(PacketFlags a, PacketFlags b) {
    return (static_cast<uint8_t>(a) & static_cast<uint8_t>(b)) != 0;
}

#pragma pack(push, 1)
struct NetworkPacketHeader {
    uint32_t sequence;
    uint32_t tick;
    uint16_t payloadSize;
    uint8_t channel;
    uint8_t flags;
};
#pragma pack(pop)

struct alignas(16) NetEntityState {
    uint32_t entityId;
    float posX, posY, posZ;
    float velX, velY, velZ;
    uint32_t lastProcessedTick;
};

// Fast 64-bit FNV-1a Hash for bit-exact network replay verification
inline uint64_t HashNetworkState(const std::vector<NetEntityState>& entities) {
    uint64_t hash = 14695981039346656037ULL;
    for (const auto& e : entities) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&e);
        for (size_t i = 0; i < sizeof(NetEntityState); ++i) {
            hash ^= ptr[i];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

// Memory-backed transport interface for headless testability & benchmarking
class INetworkTransport {
public:
    virtual ~INetworkTransport() = default;
    virtual bool Send(uint8_t channel, const void* data, size_t size, PacketFlags flags) = 0;
    virtual size_t Receive(void* outBuffer, size_t maxBufferSize, uint8_t& outChannel) = 0;
    virtual void SimulateTick(uint32_t currentTick) = 0;
    virtual const char* GetName() const = 0;
    virtual size_t GetHeaderOverheadBytes() const = 0;
};

// 1. ENet Architecture Simulation (Lightweight reliable/unreliable channels)
class ENetSimulatedTransport : public INetworkTransport {
public:
    ENetSimulatedTransport() : m_sendSeq(0) {}

    bool Send(uint8_t channel, const void* data, size_t size, PacketFlags flags) override {
        PacketItem item;
        item.header.sequence = m_sendSeq++;
        item.header.tick = m_currentTick;
        item.header.payloadSize = static_cast<uint16_t>(size);
        item.header.channel = channel;
        item.header.flags = static_cast<uint8_t>(flags);
        item.payload.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
        
        m_queue.push(std::move(item));
        return true;
    }

    size_t Receive(void* outBuffer, size_t maxBufferSize, uint8_t& outChannel) override {
        if (m_queue.empty()) return 0;

        auto item = std::move(m_queue.front());
        m_queue.pop();

        outChannel = item.header.channel;
        size_t copySize = std::min(maxBufferSize, item.payload.size());
        std::memcpy(outBuffer, item.payload.data(), copySize);
        return copySize;
    }

    void SimulateTick(uint32_t currentTick) override {
        m_currentTick = currentTick;
    }

    const char* GetName() const override { return "ENet (Simple UDP Channels)"; }
    size_t GetHeaderOverheadBytes() const override { return 12; } // Standard ENet header footprint

private:
    struct PacketItem {
        NetworkPacketHeader header;
        std::vector<uint8_t> payload;
    };
    uint32_t m_sendSeq = 0;
    uint32_t m_currentTick = 0;
    std::queue<PacketItem> m_queue;
};

// 2. GameNetworkingSockets (GNS) Simulation (Encryption + Bandwidth Management + Fragment Streams)
class GNSSimulatedTransport : public INetworkTransport {
public:
    GNSSimulatedTransport() : m_sendSeq(0) {}

    bool Send(uint8_t channel, const void* data, size_t size, PacketFlags flags) override {
        // GNS adds encryption tag (AES-GCM / libsodium 16-byte tag) + segment sequence overhead
        PacketItem item;
        item.header.sequence = m_sendSeq++;
        item.header.tick = m_currentTick;
        item.header.payloadSize = static_cast<uint16_t>(size);
        item.header.channel = channel;
        item.header.flags = static_cast<uint8_t>(flags | PacketFlags::Encrypted);

        // Add 16-byte synthetic encryption tag
        item.payload.resize(size + 16);
        std::memcpy(item.payload.data(), data, size);
        std::memset(item.payload.data() + size, 0xCC, 16); // Encryption auth tag

        m_queue.push(std::move(item));
        return true;
    }

    size_t Receive(void* outBuffer, size_t maxBufferSize, uint8_t& outChannel) override {
        if (m_queue.empty()) return 0;

        auto item = std::move(m_queue.front());
        m_queue.pop();

        outChannel = item.header.channel;
        size_t rawPayloadSize = item.payload.size() - 16; // Strip auth tag
        size_t copySize = std::min(maxBufferSize, rawPayloadSize);
        std::memcpy(outBuffer, item.payload.data(), copySize);
        return copySize;
    }

    void SimulateTick(uint32_t currentTick) override {
        m_currentTick = currentTick;
    }

    const char* GetName() const override { return "GameNetworkingSockets (Encrypted+NAT)"; }
    size_t GetHeaderOverheadBytes() const override { return 36; } // GNS connection/encryption overhead

private:
    struct PacketItem {
        NetworkPacketHeader header;
        std::vector<uint8_t> payload;
    };
    uint32_t m_sendSeq = 0;
    uint32_t m_currentTick = 0;
    std::queue<PacketItem> m_queue;
};

// 3. Raw UDP Packet Recorder & Deterministic Replay Engine
class DeterministicReplayTransport : public INetworkTransport {
public:
    struct RecordedFrame {
        uint32_t tick;
        uint32_t sequence;
        uint8_t channel;
        std::vector<uint8_t> payload;
    };

    DeterministicReplayTransport() = default;

    void SetModeRecord(bool record) { m_isRecording = record; }
    void SetModeReplay(bool replay) { m_isReplaying = replay; }

    bool Send(uint8_t channel, const void* data, size_t size, PacketFlags flags) override {
        if (m_isReplaying) return false; // In replay mode, inputs/packets are fed from recorded stream

        RecordedFrame frame;
        frame.tick = m_currentTick;
        frame.sequence = m_seq++;
        frame.channel = channel;
        frame.payload.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);

        if (m_isRecording) {
            m_recordedStream.push_back(frame);
        }
        m_liveBuffer.push(frame);
        return true;
    }

    size_t Receive(void* outBuffer, size_t maxBufferSize, uint8_t& outChannel) override {
        if (m_isReplaying) {
            if (m_replayIndex >= m_recordedStream.size()) return 0;

            const auto& frame = m_recordedStream[m_replayIndex];
            if (frame.tick > m_currentTick) return 0; // Packet not yet arrived for this tick

            outChannel = frame.channel;
            size_t copySize = std::min(maxBufferSize, frame.payload.size());
            std::memcpy(outBuffer, frame.payload.data(), copySize);
            m_replayIndex++;
            return copySize;
        }

        if (m_liveBuffer.empty()) return 0;
        auto frame = m_liveBuffer.front();
        m_liveBuffer.pop();

        outChannel = frame.channel;
        size_t copySize = std::min(maxBufferSize, frame.payload.size());
        std::memcpy(outBuffer, frame.payload.data(), copySize);
        return copySize;
    }

    void SimulateTick(uint32_t currentTick) override {
        m_currentTick = currentTick;
    }

    void ResetReplay() {
        m_replayIndex = 0;
        m_currentTick = 0;
    }

    const std::vector<RecordedFrame>& GetRecordedStream() const { return m_recordedStream; }
    const char* GetName() const override { return "Raw UDP + Deterministic Replay Stream"; }
    size_t GetHeaderOverheadBytes() const override { return 8; } // Bare UDP header

private:
    bool m_isRecording = false;
    bool m_isReplaying = false;
    uint32_t m_currentTick = 0;
    uint32_t m_seq = 0;
    size_t m_replayIndex = 0;
    std::vector<RecordedFrame> m_recordedStream;
    std::queue<RecordedFrame> m_liveBuffer;
};

} // namespace Type0
