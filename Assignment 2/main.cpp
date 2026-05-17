#include <iostream>
#include <vector>
#include <chrono>

// Include Google Highway
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

void SimdAddVectors(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& result) {
    const size_t size = a.size();
    
    // Step 1: Define the tag for scalable floats
    const hn::ScalableTag<float> d;
    
    // Step 2: Determine the number of SIMD lanes available on this CPU
    const size_t lanes = hn::Lanes(d);
    
    std::cout << "Detected SIMD Lane Width: " << lanes << " floats per vector." << std::endl;

    size_t i = 0;
    
    // Step 3: Write the vectorized loop
    for (; i + lanes <= size; i += lanes) {

        // TODO: Load 'lanes' elements from 'a' into a SIMD vector using hn::LoadU
        // Example: auto vecA = hn::LoadU(d, a.data() + i);
        auto vecA = hn::LoadU(d, a.data() + i);
        
        // TODO: Load 'lanes' elements from 'b' into a SIMD vector using hn::LoadU
        auto vecB = hn::LoadU(d, b.data() + i);
        
        // TODO: Add them together using hn::Add
        auto vecResult = hn::Add(vecA, vecB);
        
        // TODO: Store the result vector back into 'result' using hn::StoreU
        hn::StoreU(vecResult, d, result.data() + i);
    }
    
    // Step 4: Handle the remaining elements that didn't fill a complete SIMD vector
    // TODO: Write a standard scalar for-loop from `i` to `size` that does `result[j] = a[j] + b[j]`
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}


int main() {
    std::cout << "--- Assignment 2: Intro to Highway SIMD ---" << std::endl;
    
    const size_t elementCount = 1000000;
    std::vector<float> vectorA(elementCount, 1.5f);
    std::vector<float> vectorB(elementCount, 2.5f);
    std::vector<float> result(elementCount, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    
    // Call our SIMD function
    SimdAddVectors(vectorA, vectorB, result);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Processed " << elementCount << " elements in " << duration.count() << " ms." << std::endl;
    
    // Verify results
    bool correct = true;
    for (size_t i = 0; i < elementCount; ++i) {
        if (result[i] != 4.0f) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": expected 4.0, got " << result[i] << std::endl;
            break;
        }
    }

    if (correct) {
        std::cout << "SUCCESS! All calculations are correct." << std::endl;
    }

    return 0;
}
