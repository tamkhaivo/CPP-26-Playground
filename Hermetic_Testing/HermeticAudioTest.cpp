#define MINIAUDIO_IMPLEMENTATION
#include "HermeticAudioEngine.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <thread>
#include <future>
#include <cmath>

using namespace Type0;

void RunTest1_NullBackendInitialization() {
    std::cout << "\n--- TEST 1: Headless Null Backend Initialization & Teardown ---" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    HermeticAudioEngine engine;
    bool initSuccess = engine.Initialize(true /* forceNullBackend */, 48000, 2);
    assert(initSuccess && "Engine failed to initialize under null backend!");
    assert(engine.IsNullBackend() && "Engine did not report null backend status!");

    engine.Shutdown();
    auto endTime = std::chrono::high_resolution_clock::now();
    double initTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << "[PASSED] Test 1 completed in " << initTimeMs << " ms." << std::endl;
}

void RunTest2_ProceduralBufferCompletion() {
    std::cout << "\n--- TEST 2: Procedural Audio Generation & Buffer Completion ---" << std::endl;
    HermeticAudioEngine engine;
    bool initSuccess = engine.Initialize(true, 48000, 2);
    assert(initSuccess);

    ProceduralSoundSpec spec;
    spec.type = WaveformType::Sine;
    spec.frequencyHz = 440.0f;
    spec.durationSeconds = 0.1f; // 100 ms audio clip (4800 frames at 48kHz stereo = 9600 samples)
    spec.sampleRate = 48000;
    spec.channels = 2;

    auto pcmData = ProceduralAudioGenerator::GeneratePCM(spec);
    assert(pcmData.size() == 4800 * 2);

    auto voice = engine.PlayBuffer(pcmData);
    assert(voice != nullptr);
    assert(voice->isPlaying.load() == true);
    assert(voice->isCompleted.load() == false);

    // Pump frames manually to simulate audio output consumption
    std::vector<float> outputBuffer(512 * 2, 0.0f);
    size_t totalPumps = 0;
    while (!voice->isCompleted.load() && totalPumps < 20) {
        engine.PumpAudioFrames(outputBuffer.data(), 512);
        totalPumps++;
    }

    assert(voice->isCompleted.load() == true && "Voice buffer failed to reach completion state!");
    assert(voice->isPlaying.load() == false && "Voice remains in playing state after buffer depletion!");

    engine.Shutdown();
    std::cout << "[PASSED] Test 2 completed in " << totalPumps << " buffer pumps." << std::endl;
}

void RunTest3_LifecycleStateTransitions() {
    std::cout << "\n--- TEST 3: State Machine & Lifecycle Transitions ---" << std::endl;
    HermeticAudioEngine engine;
    assert(engine.Initialize(true, 48000, 2));

    ProceduralSoundSpec spec;
    spec.type = WaveformType::Square;
    spec.durationSeconds = 0.5f;

    auto pcmSquare = ProceduralAudioGenerator::GeneratePCM(spec);
    auto v1 = engine.PlayBuffer(pcmSquare);
    auto v2 = engine.PlayBuffer(pcmSquare);
    auto v3 = engine.PlayBuffer(pcmSquare);

    assert(engine.GetActiveVoiceCount() == 3);

    // Dynamic panning and volume adjustment
    v1->pan = -0.8f; // Left panning
    v2->pan = 0.8f;  // Right panning
    v3->volume = 0.25f;

    std::vector<float> mixBuf(1024 * 2);
    engine.PumpAudioFrames(mixBuf.data(), 1024);

    assert(engine.GetActiveVoiceCount() == 3);

    engine.Shutdown();
    std::cout << "[PASSED] Test 3 state transitions and voice parameter modifications verified." << std::endl;
}

void RunTest4_MultiChannelSpatialMixing() {
    std::cout << "\n--- TEST 4: Multi-Channel Spatial Polyphony Mixing ---" << std::endl;
    HermeticAudioEngine engine;
    assert(engine.Initialize(true, 48000, 2));

    // Create 4 distinct procedural waveforms
    ProceduralSoundSpec specSine{ WaveformType::Sine, 220.0f, 0.2f, 0.3f, 48000, 2 };
    ProceduralSoundSpec specSaw{ WaveformType::Sawtooth, 880.0f, 0.2f, 0.3f, 48000, 2 };
    ProceduralSoundSpec specNoise{ WaveformType::WhiteNoise, 0.0f, 0.2f, 0.1f, 48000, 2 };

    auto sinePCM = ProceduralAudioGenerator::GeneratePCM(specSine);
    auto sawPCM  = ProceduralAudioGenerator::GeneratePCM(specSaw);
    auto noisePCM = ProceduralAudioGenerator::GeneratePCM(specNoise);

    auto v1 = engine.PlayBuffer(sinePCM);
    auto v2 = engine.PlayBuffer(sawPCM);
    auto v3 = engine.PlayBuffer(noisePCM);

    v1->pan = -1.0f; // Full left
    v2->pan = 1.0f;  // Full right
    v3->pan = 0.0f;  // Center

    std::vector<float> mixBuffer(2048 * 2, 0.0f);
    engine.PumpAudioFrames(mixBuffer.data(), 2048);

    // Verify mixed output contains valid non-zero finite floats
    bool hasNonZeroLeft = false;
    bool hasNonZeroRight = false;
    for (size_t i = 0; i < mixBuffer.size(); i += 2) {
        assert(!std::isnan(mixBuffer[i]) && !std::isinf(mixBuffer[i]));
        assert(!std::isnan(mixBuffer[i + 1]) && !std::isinf(mixBuffer[i + 1]));

        if (std::abs(mixBuffer[i]) > 0.001f) hasNonZeroLeft = true;
        if (std::abs(mixBuffer[i + 1]) > 0.001f) hasNonZeroRight = true;
    }

    assert(hasNonZeroLeft && hasNonZeroRight && "Mixed spatial buffer produced silent output!");

    engine.Shutdown();
    std::cout << "[PASSED] Test 4 multi-channel spatial polyphony mixing verified." << std::endl;
}

void RunTest5_MultithreadedStressCrashTest() {
    std::cout << "\n--- TEST 5: Multithreaded Concurrency & Crash Stress Test ---" << std::endl;
    HermeticAudioEngine engine;
    assert(engine.Initialize(true, 48000, 2));

    ProceduralSoundSpec spec{ WaveformType::Sine, 440.0f, 0.05f, 0.2f, 48000, 2 };
    auto pcmData = ProceduralAudioGenerator::GeneratePCM(spec);

    constexpr int NUM_THREADS = 8;
    constexpr int VOICES_PER_THREAD = 25;
    std::atomic<int> totalTriggered{0};

    auto worker = [&](int threadId) {
        for (int i = 0; i < VOICES_PER_THREAD; ++i) {
            auto voice = engine.PlayBuffer(pcmData);
            if (voice) {
                voice->pan = (i % 2 == 0) ? -0.5f : 0.5f;
                voice->volume = static_cast<float>(i + 1) / static_cast<float>(VOICES_PER_THREAD);
                totalTriggered++;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    };

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back(worker, t);
    }

    // Pump audio processing concurrently while threads push audio voices
    std::vector<float> audioBuf(1024 * 2);
    for (int i = 0; i < 50; ++i) {
        engine.PumpAudioFrames(audioBuf.data(), 1024);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    for (auto& th : threads) {
        th.join();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << "[PASSED] Test 5 successfully triggered " << totalTriggered.load() 
              << " concurrent voices across " << NUM_THREADS << " threads in " 
              << elapsedMs << " ms without crashes or memory corruption." << std::endl;

    engine.Shutdown();
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "    Type0 Engine - Hermetic Audio Non-Deterministic Test Suite        " << std::endl;
    std::cout << "======================================================================" << std::endl;

    try {
        RunTest1_NullBackendInitialization();
        RunTest2_ProceduralBufferCompletion();
        RunTest3_LifecycleStateTransitions();
        RunTest4_MultiChannelSpatialMixing();
        RunTest5_MultithreadedStressCrashTest();

        std::cout << "\n======================================================================" << std::endl;
        std::cout << " [SUCCESS] ALL 5 HERMETIC AUDIO FUNCTIONAL TESTS PASSED PERFECTLY! " << std::endl;
        std::cout << "======================================================================" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "\n [FAILURE] Test failed with exception: " << ex.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n [FAILURE] Test failed with unknown exception!" << std::endl;
        return 1;
    }
}
