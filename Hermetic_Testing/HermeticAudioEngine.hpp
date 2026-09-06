#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include <cmath>
#include <string>
#include <iostream>
#include <thread>
#include <random>
#include <mutex>
#include <algorithm>

// Ensure miniaudio is configured safely
#ifndef MA_NO_DECODING
#define MA_NO_DECODING
#endif
#ifndef MA_NO_ENCODING
#define MA_NO_ENCODING
#endif

#include "miniaudio.h"

namespace Type0 {

enum class WaveformType {
    Sine,
    Square,
    Sawtooth,
    WhiteNoise
};

struct ProceduralSoundSpec {
    WaveformType type = WaveformType::Sine;
    float frequencyHz = 440.0f;
    float durationSeconds = 1.0f;
    float amplitude = 0.5f;
    uint32_t sampleRate = 48000;
    uint32_t channels = 2;
};

class ProceduralAudioGenerator {
public:
    static std::vector<float> GeneratePCM(const ProceduralSoundSpec& spec) {
        size_t totalFrames = static_cast<size_t>(spec.durationSeconds * spec.sampleRate);
        size_t totalSamples = totalFrames * spec.channels;
        std::vector<float> pcmBuffer(totalSamples, 0.0f);

        constexpr float TWO_PI = 6.28318530717958647692f;
        std::mt19937 rng(1337); // Deterministic seed for test procedural audio generation
        std::uniform_real_distribution<float> noiseDist(-spec.amplitude, spec.amplitude);

        for (size_t f = 0; f < totalFrames; ++f) {
            float time = static_cast<float>(f) / static_cast<float>(spec.sampleRate);
            float sampleVal = 0.0f;

            switch (spec.type) {
                case WaveformType::Sine:
                    sampleVal = spec.amplitude * std::sin(TWO_PI * spec.frequencyHz * time);
                    break;
                case WaveformType::Square:
                    sampleVal = (std::sin(TWO_PI * spec.frequencyHz * time) >= 0.0f) ? spec.amplitude : -spec.amplitude;
                    break;
                case WaveformType::Sawtooth:
                    sampleVal = spec.amplitude * (2.0f * (spec.frequencyHz * time - std::floor(spec.frequencyHz * time + 0.5f)));
                    break;
                case WaveformType::WhiteNoise:
                    sampleVal = noiseDist(rng);
                    break;
            }

            for (uint32_t c = 0; c < spec.channels; ++c) {
                pcmBuffer[f * spec.channels + c] = sampleVal;
            }
        }

        return pcmBuffer;
    }
};

struct SoundVoiceState {
    uint32_t voiceId = 0;
    std::atomic<bool> isPlaying{false};
    std::atomic<bool> isCompleted{false};
    std::atomic<float> volume{1.0f};
    std::atomic<float> pan{0.0f}; // -1.0 (left) to +1.0 (right)
    size_t currentFrameIndex = 0;
    std::vector<float> pcmData;
    uint32_t channels = 2;
};

class HermeticAudioEngine {
public:
    HermeticAudioEngine() = default;
    ~HermeticAudioEngine() {
        Shutdown();
    }

    bool Initialize(bool forceNullBackend = true, uint32_t sampleRate = 48000, uint32_t channels = 2) {
        m_sampleRate = sampleRate;
        m_channels = channels;
        m_nullBackendEnabled = forceNullBackend;

        ma_context_config contextConfig = ma_context_config_init();
        ma_backend backends[] = { ma_backend_null, ma_backend_wasapi, ma_backend_dsound };
        uint32_t backendCount = forceNullBackend ? 1 : 3;

        if (ma_context_init(forceNullBackend ? backends : nullptr, forceNullBackend ? backendCount : 0, &contextConfig, &m_context) != MA_SUCCESS) {
            std::cerr << "[HermeticAudioEngine] Failed to initialize miniaudio context." << std::endl;
            return false;
        }

        ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
        deviceConfig.playback.format   = ma_format_f32;
        deviceConfig.playback.channels = m_channels;
        deviceConfig.sampleRate        = m_sampleRate;
        deviceConfig.dataCallback      = AudioDataCallbackStatic;
        deviceConfig.pUserData         = this;

        if (ma_device_init(&m_context, &deviceConfig, &m_device) != MA_SUCCESS) {
            std::cerr << "[HermeticAudioEngine] Failed to initialize miniaudio device." << std::endl;
            ma_context_uninit(&m_context);
            return false;
        }

        if (ma_device_start(&m_device) != MA_SUCCESS) {
            std::cerr << "[HermeticAudioEngine] Failed to start miniaudio device." << std::endl;
            ma_device_uninit(&m_device);
            ma_context_uninit(&m_context);
            return false;
        }

        m_initialized = true;
        std::cout << "[HermeticAudioEngine] Initialized successfully. Backend: " 
                  << (forceNullBackend ? "ma_backend_null (Headless Hermetic)" : "Hardware/Default OS") 
                  << " | Sample Rate: " << m_sampleRate << " Hz | Channels: " << m_channels << std::endl;
        return true;
    }

    void Shutdown() {
        if (!m_initialized) return;
        m_initialized = false;

        ma_device_stop(&m_device);
        ma_device_uninit(&m_device);
        ma_context_uninit(&m_context);

        std::lock_guard<std::mutex> lock(m_voicesMutex);
        m_voices.clear();
        std::cout << "[HermeticAudioEngine] Engine shut down cleanly." << std::endl;
    }

    std::shared_ptr<SoundVoiceState> PlayBuffer(const std::vector<float>& pcmData, bool loop = false) {
        if (!m_initialized || pcmData.empty()) return nullptr;

        auto voice = std::make_shared<SoundVoiceState>();
        voice->voiceId = m_nextVoiceId++;
        voice->pcmData = pcmData;
        voice->channels = m_channels;
        voice->isPlaying = true;
        voice->isCompleted = false;

        {
            std::lock_guard<std::mutex> lock(m_voicesMutex);
            m_voices.push_back(voice);
        }

        return voice;
    }

    size_t GetActiveVoiceCount() {
        std::lock_guard<std::mutex> lock(m_voicesMutex);
        size_t count = 0;
        for (const auto& v : m_voices) {
            if (v->isPlaying.load()) count++;
        }
        return count;
    }

    bool IsNullBackend() const { return m_nullBackendEnabled; }

    // Direct manual audio buffer pump for hermetic step testing without waiting for wall-clock device threads
    void PumpAudioFrames(float* pOutput, uint32_t frameCount) {
        ProcessAudio(pOutput, frameCount);
    }

private:
    static void AudioDataCallbackStatic(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
        auto* engine = static_cast<HermeticAudioEngine*>(pDevice->pUserData);
        if (engine) {
            engine->ProcessAudio(static_cast<float*>(pOutput), frameCount);
        }
    }

    void ProcessAudio(float* pOutput, uint32_t frameCount) {
        size_t totalSamples = frameCount * m_channels;
        std::fill_n(pOutput, totalSamples, 0.0f);

        std::lock_guard<std::mutex> lock(m_voicesMutex);
        for (auto it = m_voices.begin(); it != m_voices.end();) {
            auto& voice = *it;

            if (!voice->isPlaying.load()) {
                ++it;
                continue;
            }

            size_t totalVoiceFrames = voice->pcmData.size() / voice->channels;
            size_t framesRemaining = totalVoiceFrames - voice->currentFrameIndex;
            size_t framesToMix = std::min<size_t>(frameCount, framesRemaining);

            float vol = voice->volume.load();
            float pan = voice->pan.load(); // -1.0 = left, +1.0 = right
            float leftGain  = vol * std::cos((pan + 1.0f) * 0.25f * 3.14159265f);
            float rightGain = vol * std::sin((pan + 1.0f) * 0.25f * 3.14159265f);

            for (size_t f = 0; f < framesToMix; ++f) {
                size_t srcFrame = voice->currentFrameIndex + f;
                size_t dstFrame = f;

                float leftSample = voice->pcmData[srcFrame * voice->channels] * leftGain;
                float rightSample = (voice->channels > 1 ? voice->pcmData[srcFrame * voice->channels + 1] : leftSample) * rightGain;

                pOutput[dstFrame * m_channels] += leftSample;
                if (m_channels > 1) {
                    pOutput[dstFrame * m_channels + 1] += rightSample;
                }
            }

            voice->currentFrameIndex += framesToMix;
            if (voice->currentFrameIndex >= totalVoiceFrames) {
                voice->isPlaying = false;
                voice->isCompleted = true;
            }

            ++it;
        }
    }

    ma_context m_context{};
    ma_device m_device{};
    uint32_t m_sampleRate = 48000;
    uint32_t m_channels = 2;
    bool m_initialized = false;
    bool m_nullBackendEnabled = true;

    std::mutex m_voicesMutex;
    std::vector<std::shared_ptr<SoundVoiceState>> m_voices;
    std::atomic<uint32_t> m_nextVoiceId{1};
};

} // namespace Type0
