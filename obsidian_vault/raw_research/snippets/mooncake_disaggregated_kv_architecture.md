# Mooncake: KVCache-Centric Disaggregated Serving & Hierarchical Memory (Zhong et al., FAST 2025)

## 1. The Prefill-Decode Disaggregation Paradigm
Standard LLM serving systems colocate compute-intensive prompt prefilling ($\mathcal{O}(L_{\text{prompt}}^2)$ FLOPs) with memory-bandwidth-intensive autoregressive token decoding ($\mathcal{O}(1)$ arithmetic intensity). This hardware interference creates severe pipeline preemption, spiking time-between-tokens (TBT) and violating latency Service Level Objectives (SLOs).
The Mooncake architecture (Zhong et al., Moonshot AI / Kimi, FAST 2025 Best Paper / arXiv:2407.00079) separates inference into dedicated **Prefill** (high-compute Tensor Core saturation) and **Decode** (high-bandwidth PagedAttention) server pools connected by an asynchronous KV transfer fabric.

## 2. Hierarchical 3-Tier KV Memory Hierarchy
To prevent catastrophic eviction during long-context and multi-turn agent sessions, Mooncake disaggregates the KV cache into three physical storage tiers:
1. **Tier 1 (GPU HBM):** Serves active autoregressive decoding tokens ($< 1\,\mu\text{s}$ access, $2.0\text{--}3.35\,\text{TB/s}$).
2. **Tier 2 (Host CPU DRAM / CXL 2.0 Fabric):** Acts as high-capacity warm cache for active session state and shared prefix trees ($100\text{--}200\,\text{ns}$ interconnect overhead, $200\text{--}400\,\text{GB/s}$).
3. **Tier 3 (Distributed NVMe SSD Fabric):** Stores persistent cold historical conversational context, completely eliminating prefill recomputation on resume.

## 3. Transfer Engine & Production Metrics
- **Zero-Copy RDMA Engine:** Materialized KV tensors are transferred directly between nodes via RoCEv2, requiring only $\sim 64\,\text{ms}$ for a $128\text{k}$-token context across $400\,\text{Gbps}$ links.
- **Conductor Scheduler:** Routes incoming requests based on KV cache locality and employs prediction-based early rejection to survive sudden traffic spikes.
- **Empirical Scale:** Achieves up to **525% throughput improvement** on long contexts and allows Kimi production clusters to handle **75% higher request loads** under strict latency constraints.
