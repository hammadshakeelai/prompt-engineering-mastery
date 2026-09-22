# LoRA & QLoRA Matrix Decomposition

- **LoRA**: Freezes base weights $W_0$ and updates weights via low-rank decomposition $\Delta W = \frac{\alpha}{r} B A$ ($r \ll d$). Merges into base weights at inference with zero added latency.
- **QLoRA**: Quantizes base weights to 4-bit NormalFloat (NF4), adds Double Quantization and Paged Optimizers, backpropagating 16-bit adapter gradients through dequantized weights.