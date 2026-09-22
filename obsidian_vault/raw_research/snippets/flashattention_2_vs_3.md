# FlashAttention-2 vs. FlashAttention-3 Architectural Optimizations

- **Hardware Targeting**: FA-2 optimized for Ampere via sequence parallelization. FA-3 targets Hopper (H100) Tensor Memory Accelerator (TMA) and hardware barriers (mbarrier) for asynchronous data movement between GMEM and SMEM.
- **Warp Specialization**: FA-3 separates warps into dedicated producer (TMA loading) and consumer (Tensor Core MMA) warps, removing FA-2 cross-warp synchronization overhead.
- **Interleaved GEMM-Softmax**: Overlaps Tensor Core matmuls with ALU softmax operations (ping-pong pipelining).
- **FP8 Support**: Native FP8 execution with block scaling doubles compute throughput over FA-2 FP16 baseline.