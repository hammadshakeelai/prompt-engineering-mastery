# Compressed FSMs & Jump-Forward Decoding (SGLang, Zheng et al. ICML 2024)

## 1. The Bottleneck of Autoregressive Boilerplate
Enforcing grammar or JSON schema constraints often generates deterministic static substrings (e.g., `{"status": "`, `", "items": [`).
- **GPU Inefficiency:** Generating deterministic tokens autoregressively forces single-token forward passes, bottlenecking execution on GPU High Bandwidth Memory (HBM) transfers.

## 2. Compressed Finite State Machine (cFSM) Formulation
Traditional FSMs maintain single-token transition arcs. The cFSM algorithm identifies deterministic execution paths:
- **Linear Path Compression:** Any sequence of states with out-degree 1:
  $$s_0 \xrightarrow{t_1} s_1 \xrightarrow{t_2} \dots \xrightarrow{t_k} s_k$$
  is collapsed into a single macro-transition:
  $$s_0 \xrightarrow{(t_1, t_2, \dots, t_k)} s_k$$

## 3. Jump-Forward Execution & RadixAttention Co-Design
1. **Jump-Forward:** When entering state $s_0$, the inference engine appends the entire multi-token sequence $(t_1, \dots, t_k)$ to the prompt buffer, skipping $k$ autoregressive decode iterations.
2. **Chunked Prefill Verification:** Instead of $k$ separate memory loads, the engine processes all jumped tokens in one batched chunked prefill operation.
3. **RadixAttention Cache Hit:** Pre-computed schema structures are cached in Radix trees, avoiding recomputation.
4. **Empirical Acceleration:** Achieves $2\times\text{--}5\times$ speedups in structured generation tasks while guaranteeing zero syntactic syntax errors.
