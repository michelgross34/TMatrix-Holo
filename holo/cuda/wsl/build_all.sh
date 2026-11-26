#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/usr/local/cuda/bin:$PATH"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DHOLO_MODE=CPU_OPENMP_CUDA
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
