#cuda #hardware #architecture

# 1.  Overview

## 1.1. Streaming Multiprocessor

- NVidia GPU architecture are built around a scalable array of multithreaded __streaming multiprocessors__ (SMs).
- Each SM can be thought of like a core in a CPU.
- Unlike a core in a CPU, a SM is designed to work with many threads at a times (can be thousands of threads)

## 1.2. Threads

- A thread is an unit of execution
- Similar to a thread executed CPU, a GPU thread also has stack, registers, access to memory
- Unlike CPU thread, GPU threads are grouped into blocks
- Each block run on 1 SM at a time
- Many blocks can run concurrently on a SM

## 1.3. Warps

- A block is sub-divided into warps. Each warp contains 32 threads.
- Each warp is an unit of scheduling
- Every threads on a warp execute the same instruction at any time. If branching occurs, all threads go to the branch execute with the other threads are suspended. ==NOTE==: branch divergence only occurs in a warp. Different warps may execute independently.