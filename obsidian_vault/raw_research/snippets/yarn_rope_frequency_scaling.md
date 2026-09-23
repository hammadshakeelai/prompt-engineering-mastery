# YaRN: Frequency-Band RoPE Scaling & Attention Entropy Calibration (Peng et al., ICLR 2024)

## 1. Failure Modes of Linear Position Interpolation
Direct Position Interpolation (PI: $m' = m / s$) compresses all Fourier frequencies uniformly, corrupting high-frequency rotational dynamics and degrading short-range syntax and token boundary awareness.

## 2. Frequency-Band Partitioning Formulation
YaRN (Peng et al., ICLR 2024 / arXiv:2309.00071) divides embedding dimensions by wavelength ratio $r_i = \frac{L \theta_i}{2\pi}$:
- **High-Frequency Band ($r_i > \beta$):** Zero interpolation ($\theta'_i = \theta_i$). Preserves exact local relative order and syntax.
- **Low-Frequency Band ($r_i < \alpha$):** Full linear interpolation ($\theta'_i = \theta_i / s$). Maps macroscopic context distances.
- **Mid-Frequency Band ($\alpha \le r_i \le \beta$):** Smooth ramp function:
  $$\gamma(r_i) = \frac{\beta - r_i}{\beta - \alpha}, \quad \theta'_i = (1 - \gamma) \theta_i + \gamma \frac{\theta_i}{s}$$
  where $\alpha = 1, \beta = 32$.

## 3. Attention Entropy Calibration
Extending context length by $s = 32\times$ artificially disperses attention softmax probability masses across $32\times$ more keys. YaRN rescales the attention temperature by factor $\sqrt{t}$:
$$\operatorname{Attention}(Q, K, V) = \operatorname{Softmax}\left( \frac{Q K^\top}{\sqrt{d_k} \sqrt{t}} \right) V, \quad \sqrt{t} = \sqrt{0.1 \ln(s) + 1}$$
preserving attention sharpness across 128k contexts.

## 4. Empirical Efficiency
Enables extending Llama-2 from 4k to 128k context with **$10\times$ fewer tokens** and **$2.5\times$ fewer training steps** than standard methods, with zero performance loss on short-context tasks.
