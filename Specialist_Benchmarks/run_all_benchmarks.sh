#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "================================================================================"
echo "RUNNING ALL SPECIALIST BENCHMARKS (Concurrency, ECS, Scripting, Networking)"
echo "Host CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs) (32 Threads)"
echo "Date/Time: $(date -u)"
echo "================================================================================"
echo ""

echo ">>> RUNNING BENCHMARK 1: TASK SCHEDULERS & CONCURRENCY <<<"
./bench_concurrency
echo ""

echo ">>> RUNNING BENCHMARK 2: ECS (FLECS v4 VS ENTT v3) <<<"
./bench_ecs
echo ""

echo ">>> RUNNING BENCHMARK 3: SCRIPTING & SANDBOXING (LUAU VS WASM3) <<<"
./bench_scripting
echo ""

echo ">>> RUNNING BENCHMARK 4: NETWORK & TRANSPORT (GNS VS LIBDATACHANNEL VS YOJIMBO) <<<"
./bench_networking
echo ""

echo "================================================================================"
echo "ALL SPECIALIST BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
