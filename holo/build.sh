#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Prepare private copies without modifying the neighbouring Mishchenko project.
mkdir -p build
python3 prepare_cmake.py build/generated
flags=(-std=legacy -ffixed-line-length-none -fallow-argument-mismatch -O2 -w -Ibuild/generated/build_src/fixed_dp)
gfortran "${flags[@]}" -c build/generated/holo_core.f -o build/holo_core.o
gfortran "${flags[@]}" -c build/generated/build_src/fixed_dp/lpd.f -o build/lpd.o
gfortran "${flags[@]}" -Ibuild/generated/build_src/fixed_qp -c build/generated/holo_qp.f -o build/holo_qp.o
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic holo.cpp build/holo_core.o build/lpd.o build/holo_qp.o -lfftw3 -lgfortran -lquadmath -o mish_holo
echo "Built $(pwd)/mish_holo"
