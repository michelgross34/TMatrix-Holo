#!/usr/bin/env bash
# Configure, compile and test HOLO (WSL, Release).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
case "${1:-all}" in
  cpu) mode=CPU; build=build_cpu ;;
  openmp) mode=CPU_OPENMP; build=build_openmp ;;
  cuda|both|all) mode=CPU_OPENMP_CUDA; build=build ;;
  *) echo "Usage: $0 [cpu|openmp|cuda|all]" >&2; exit 2 ;;
esac
if [[ "$mode" == CPU_OPENMP_CUDA ]]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi
echo '=== Configuring HOLO WSL ==='
cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DHOLO_MODE="$mode"
echo '=== Building all targets ==='
cmake --build "$build" --parallel
echo '=== Running tests ==='
ctest --test-dir "$build" --output-on-failure
echo 'Build and tests successful.'
printf 'CPU:    %s/%s/mish_holo_cpu\n' "$PWD" "$build"
if [[ "$mode" != CPU ]]; then printf 'OpenMP: %s/%s/mish_holo_openmp\n' "$PWD" "$build"; fi
if [[ "$mode" == CPU_OPENMP_CUDA ]]; then printf 'CUDA: %s/%s/mish_holo_cuda\n' "$PWD" "$build"; fi
printf 'Results: %s/%s/run\n' "$PWD" "$build"
