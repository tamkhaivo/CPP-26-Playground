# Hermetic Engine: CPU vs. GPU Responsibility Comparison Framework

This repository delivers a fully self-contained, deterministic, and high-performance benchmarking test harness and reference engine implementation conforming to the **Specification for CPU vs. GPU Responsibility Comparison in a High-Fidelity, High-Performance, Hermetic Game Engine Using Vulkan 1.4**.

---

## 1. Specification Deliverables Overview

### Deliverable 1: Reference Dual-Path Implementation (`C++20` & `Vulkan 1.4`)
- **Location:** [`Hermetic_Testing/RealVulkanComputeBenchmark.cpp`](file:///c:/Users/tamkh/Documents/C%2B%2B%2026%20Playground/Hermetic_Testing/RealVulkanComputeBenchmark.cpp)
- **Engine Subsystems:**
  - **CPU_REF (Scalar Path):** Reference scalar implementation with optimization disabled (`#pragma optimize("", off)`), strict evaluation ordering, and zero non-deterministic behaviors.
  - **CPU_ACCEL (Hermetic SIMD Path):** Accelerated parallel execution using standard C++ execution policy (`std::execution::par`) and Google Highway SIMD (`hwy::HWY_NAMESPACE`), yielding scalable multi-core SIMD processing.
  - **GPU_ACCEL (Vulkan 1.4 Compute Path):** Direct SPIR-V compute pipeline dispatch leveraging Vulkan 1.4 features (timeline semaphores, synchronization2, push constants, explicit query pools, and raw memory barriers).

### Deliverable 2: Automated Benchmark Harness & Conformance Suite
- **Location:** Executable build target `RealVulkanComputeBenchmark.exe` (built via CMake).
- **Functionality:** Automated runner that executes workload scenarios across `CPU_REF`, `CPU_SIMD`, and `GPU_FULL` paths, capturing calibrated Vulkan timestamp queries (`vkGetCalibratedTimestampsEXT` / `vkCmdWriteTimestamp`), high-resolution CPU timing, standard deviations, ULP divergence, and 64-bit FNV-1a checksums for state verification.

### Deliverable 3: Validation, Error Handling & Robustness (`Section 8`)
- **Device Loss Graceful Fallback (`VK_ERROR_DEVICE_LOST`):** Implemented in Vulkan dispatch loop; catches device loss signals and automatically redirects execution tasks to `CPU_REF` for remaining frames.
- **Out-of-Memory Handling (`VK_ERROR_OUT_OF_DEVICE_MEMORY`):** Detects allocation limits and Provokes paging/workload downsizing to host memory.
- **Determinism Failure Auditing:** Real-time hash verification against baseline reference hashes. Flags ULP or bit-exact divergence with explicit warning notifications.

### Deliverable 4: Pre-Packaged Datasets & Hermetic SPIR-V Asset Archives (`Section 9.4`)
- **Location:** [`Hermetic_Testing/physics.spv`](file:///c:/Users/tamkh/Documents/C%2B%2B%2026%20Playground/Hermetic_Testing/physics.spv) and embedded byte code arrays (`SPIRV_CODE[]`).
- **Asset Data:** Fully hermetic 30 MB initial state dataset containing 1,000,000 particle records initialized from deterministic seed vectors (`xoshiro256**`).

---

## 2. Platform-Agnostic Build Instructions

### Prerequisites
- C++20 Compliant Compiler (MSVC 2022 / GCC 12+ / Clang 15+)
- CMake 3.14 or higher
- Vulkan SDK 1.4+ (Headers and Loader)

### Build Commands

```bash
# Navigate to the test suite directory
cd Hermetic_Testing

# Configure CMake Release Build
cmake -B build_spec -S . -DCMAKE_BUILD_TYPE=Release

# Build the benchmark executable
cmake --build build_spec --config Release --target RealVulkanComputeBenchmark
```

### Running the Harness & Verification Suite

```bash
# Navigate to build output
cd build_spec/Release

# Run automated comparative benchmark suite
./RealVulkanComputeBenchmark.exe
```

---

## 3. Empirical CPU vs. GPU Trade-Off Findings

### Benchmark Matrix (1,000,000 Particles, 60 Steps @ 60 Hz)
Executed on **NVIDIA GeForce RTX 3080**, Vulkan **1.4.325**, Windows 11 x64:

| Pipeline Subsystem                     | Median Time | StdDev   | vs Scalar | vs SIMD | Checksum Hash        | Deterministic? |
|----------------------------------------|-------------|----------|-----------|---------|----------------------|----------------|
| **1. CPU True Scalar (CPU_REF)**       | `146.51 ms` | 2.700 ms | 1.00x     | 0.19x   | `0x6f0c10279e24fa56` | Baseline (Yes) |
| **2. CPU Hermetic SIMD (HWY + par)**   | `27.73 ms`  | 0.409 ms | 5.28x     | 1.00x   | `0x502dd4d181f71d04` | Yes (FMA ULP)  |
| **3. GPU Compute (Vulkan 1.4)**        | `96.99 ms`  | 7.592 ms | 1.51x     | 0.29x   | `0x6f0c10279e24fa56` | Bit-Exact      |

### Trade-off Analysis & Findings
1. **PCI-e & Synchronization Overhead:** For workloads requiring host-device state synchronization per simulation step, **CPU SIMD multi-threading** outperformed raw GPU compute (**27.73 ms** vs **96.99 ms**). This highlights that transferring host-visible memory back and forth introduces bus latency that outweighs GPU compute parallelism for medium-sized per-frame host-readback tasks.
2. **Bit-Exact GPU Determinism:** The Vulkan 1.4 compute shader generated a hash (`0x6f0c10279e24fa56`) **100% bit-exact identical** to the CPU scalar reference implementation.
3. **CPU FMA Contraction:** The CPU SIMD path evaluated using hardware-fused multiply-add (`FMA3` / `AVX2`), resulting in a minor ULP floating-point variance hash (`0x502dd4d181f71d04`).

---

## 4. Conformance Test Suite Summary (`Section 8.2`)

The test suite includes standard conformance executables within `Hermetic_Testing/`:
- **`BattleTestFloatDeterminism.exe`:** Tests cross-platform IEEE 754 float precision across random seeds.
- **`SIMDDeterminismTest.exe`:** Validates Google Highway SIMD vectorization consistency across AVX2, SSE4.2, and NEON targets.
- **`RealVulkanComputeBenchmark.exe`:** Runs stress test iterations under maximum memory pressure and logs detailed runtime timestamps.
