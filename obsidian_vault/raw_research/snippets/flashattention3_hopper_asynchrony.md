# FlashAttention-3: Asynchronous Hopper Attention & FP8 (Shah et al., Stanford 2024)

## 1. Hopper Hardware Bottlenecks
FlashAttention-2 achieved $<35\%$ utilization of NVIDIA H100 Tensor Cores ($240\text{--}350\text{ TFLOPS}$ FP16) due to synchronous GMEM-to-register load stalls and ALU softmax bottlenecks.

## 2. Asynchronous Primitives & Warp Specialization
FlashAttention-3 (Shah et al., Stanford / Colfax 2024 / arXiv:2407.08608) introduces:
- **TMA & `mbarrier`:** Direct hardware-managed GMEM-to-SMEM async transfers bypassing registers.
- **Warp Specialization:** Producer warps issue TMA memory descriptors, while consumer warp groups execute math, decoupling latency.
- **Ping-Pong Scheduling:** Interleaves async Tensor Core GEMMs ($QK^\top$ and $PV$) with online softmax updates across alternating warp groups, completely hiding softmax ALU cycles.

## 3. FP8 with Incoherent Processing
Applies a Randomized Hadamard Transform $X \leftarrow X H$ prior to quantization, dispersing activation outlier spikes across dimensions and enabling FP8 Tensor Core execution without perplexity loss.

## 4. Benchmark Performance
- **FP16:** Achieves **$740\text{ TFLOPS}$** ($75\%\text{--}80\%$ of theoretical peak, $1.5\times\text{--}2.0\times$ speedup over FA-2).
- **FP8:** Reaches **$1.2\text{ PFLOPS}$ ($1200\text{ TFLOPS}$)** on H100 SXM5 with $2.6\times$ lower numerical error than standard baselines.
