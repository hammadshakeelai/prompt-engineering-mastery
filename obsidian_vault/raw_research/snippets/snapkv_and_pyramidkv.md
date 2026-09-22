# KV Cache Pruning: SnapKV & PyramidKV (2024)

## The Long-Context Memory Bottleneck
During long-context generation, storing Key-Value tensors across all layers consumes massive GPU VRAM ($2 \times b \times s \times l \times h \times d$), bottlenecking throughput and concurrency.

## 1. SnapKV: Observation Window Pruning (Li et al., NeurIPS 2024)
- **Core Observation:** Individual attention heads consistently focus on specific, stable prompt token clusters rather than shifting randomly across generation steps.
- **Mechanism:** Uses a small "observation window" at the end of the prompt to evaluate attention weights before autoregressive generation begins.
- **Clustered Selection:** Selects and freezes the most salient KV positions per head, evicting the rest.
- **Results:** Delivers up to 3.6x generation acceleration and 8.2x memory savings with negligible accuracy degradation; requires zero fine-tuning.

## 2. PyramidKV: Layer-Wise Funneling (Cai et al., 2024)
- **Core Observation:** Attention dynamics differ fundamentally across network depth: lower layers exhibit broad, diffuse attention across many tokens, whereas deeper layers concentrate heavily on a few critical conceptual tokens.
- **Mechanism:** Instead of allocating uniform KV cache quotas per layer, PyramidKV allocates a "pyramidal" cache budget: retaining wide KV capacities in early/lower layers and aggressively pruned, concentrated caches in deep layers.
- **Results:** Outperforms uniform budget pruning, retaining top-tier generation quality while utilizing as little as 12% of the standard KV cache footprint.
