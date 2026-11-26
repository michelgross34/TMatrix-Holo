cmake_minimum_required(VERSION 3.20)
project(Mish_Holo_CUDA LANGUAGES CXX Fortran)
# Compatibility entry point: hybrid mode by default.
set(HOLO_DEFAULT_MODE CPU_OPENMP_CUDA)
include("../../cmake/Holo.cmake")
