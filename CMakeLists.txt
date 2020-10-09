cmake_minimum_required(VERSION 3.12...3.18)

project(cuda-graphs-ubench
  VERSION 0.0.1
  LANGUAGES CUDA CXX)

set (CMAKE_CXX_STANDARD 17)

add_executable(benchmark main.cpp kernels.cu)
