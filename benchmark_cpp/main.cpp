#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

#include "TaskScheduler.h"
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

struct alignas(16) MeshInstanceData {
    float x, y, z, w;
};

void ParallelTransformPipeline(enki::TaskScheduler& ts, std::vector<MeshInstanceData>& dataBuffer, float deltaTime) {
    const size_t elementCount = dataBuffer.size();
    const size_t chunkSize = 1024;
    const size_t totalChunks = (elementCount + chunkSize - 1) / chunkSize;

    enki::TaskSet task(totalChunks, [&](enki::TaskSetPartition range, uint32_t threadnum) {
        const hn::ScalableTag<float> d;
        const auto simdDelta = hn::Set(d, deltaTime);
        const auto velocityMod = hn::Set(d, 9.81f);
        const size_t lanes = hn::Lanes(d);

        for (uint32_t chunkIdx = range.start; chunkIdx < range.end; ++chunkIdx) {
            size_t startOffset = chunkIdx * chunkSize;
            size_t endOffset = std::min(startOffset + chunkSize, elementCount);

            for (size_t i = startOffset; i < endOffset; i += lanes) {
                if (i + lanes <= endOffset) {
                    float values[16];
                    for(size_t l = 0; l < lanes; ++l) values[l] = dataBuffer[i + l].y;
                    
                    auto yValues = hn::LoadU(d, values);
                    auto updatedY = hn::MulAdd(simdDelta, velocityMod, yValues);
                    
                    hn::StoreU(updatedY, d, values);
                    for(size_t l = 0; l < lanes; ++l) dataBuffer[i + l].y = values[l];
                } else {
                    // Scalar fallback for tail
                    for (size_t j = i; j < endOffset; ++j) {
                        dataBuffer[j].y += deltaTime * 9.81f;
                    }
                }
            }
        }
    });

    ts.AddTaskSetToPipe(&task);
    ts.WaitforTask(&task);
}

int main() {
    enki::TaskScheduler ts;
    ts.Initialize();

    const size_t NUM_OBJECTS = 10'000'000;
    std::vector<MeshInstanceData> buffer(NUM_OBJECTS);
    for (size_t i = 0; i < NUM_OBJECTS; ++i) {
        buffer[i] = {0.0f, static_cast<float>(i % 100), 0.0f, 1.0f};
    }

    std::cout << "Starting C++ Benchmark (HWY + enkiTS) with " << NUM_OBJECTS << " objects.\n";

    // Warmup
    ParallelTransformPipeline(ts, buffer, 0.016f);

    auto start = std::chrono::high_resolution_clock::now();
    
    const int ITERATIONS = 100;
    for (int i = 0; i < ITERATIONS; ++i) {
        ParallelTransformPipeline(ts, buffer, 0.016f);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    std::cout << "Average time per frame: " << elapsed.count() / ITERATIONS << " ms\n";
    std::cout << "Sample value: " << buffer[500].y << "\n";

    return 0;
}
