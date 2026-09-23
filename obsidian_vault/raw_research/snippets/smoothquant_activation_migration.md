# SmoothQuant: Activation Outlier Migration (Xiao et al., ICML 2023)

## 1. The Activation Outlier Quantization Bottleneck
Quantizing Large Language Models to 8-bit integers (INT8) across both weights and activations (W8A8) unlocks fast INT8 Tensor Core matrix multiplications. However, activations suffer from persistent, channel-specific outliers ($|X_j| \gg 100$) in models $>6.7\text{B}$, which crush quantization resolution for normal activation channels.

## 2. Mathematically Equivalent Scale Migration
SmoothQuant (Guangxuan Xiao et al., ICML 2023 / arXiv:2211.10438) migrates quantization difficulty from activations to weights via an exact mathematical transformation:
$$Y = X W = \left( X \cdot \text{diag}(s)^{-1} \right) \left( \text{diag}(s) \cdot W \right) = \hat{X} \hat{W}$$

1. **Migration Scale Vector ($s \in \mathbb{R}^C$):**
   $$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$
   Setting $\alpha = 0.5$ balances the dynamic ranges symmetrically between activation channels and weight columns.

2. **Offline Weight Scaling & Online Norm Fusion:**
   - $\hat{W} = \text{diag}(s) W$ is computed and quantized to INT8 offline.
   - $\hat{X} = X \oslash s$ is fused directly into the preceding LayerNorm or RMSNorm layer without runtime overhead.

## 3. Empirical Results
- Achieves **zero perplexity degradation** under complete W8A8 quantization across OPT, BLOOM, LLaMA-1/2, and Mistral.
- Delivers **$1.56\times$ GEMM speedup** and **$2\times$ memory reduction**, enabling 530B-parameter models to be served within a single GPU node.
