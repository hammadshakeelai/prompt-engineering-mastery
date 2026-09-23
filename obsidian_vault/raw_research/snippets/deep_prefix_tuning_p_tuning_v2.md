# Deep Prefix-Tuning & Multi-Layer Continuous Steering (P-Tuning v2, Liu et al., ACL 2022)

## 1. Limitations of Input Prompt Tuning
Shallow prompt tuning (Lester et al., 2021) inserts continuous tokens only at the input layer. In deep networks ($>30$ layers) and models $\le 10\text{B}$, prompt steering influence vanishes as activations pass through successive nonlinearities, causing performance degradation on complex tasks.

## 2. Multi-Layer Key-Value Virtual Prefix Formulation
P-Tuning v2 (Liu et al., ACL 2022 / Li & Liang, ACL 2021) injects continuous prefix parameter matrices directly into the Key and Value projections of *every transformer layer* $l \in \{1, \dots, L\}$:
$$\tilde{K}^{(l)} = \left[ P_K^{(l)} \,;\, K_{\text{seq}}^{(l)} \right], \quad \tilde{V}^{(l)} = \left[ P_V^{(l)} \,;\, V_{\text{seq}}^{(l)} \right]$$
where $P_K^{(l)}, P_V^{(l)} \in \mathbb{R}^{l_p \times d}$.
- **Invariant Steering Authority:** Prefix vectors provide persistent layerwise conditioning that cannot be overwritten by upstream residual stream activations.
- **Universal Scale Invariance:** Matches full parameter fine-tuning across all model sizes (from 300M to 13B+) while updating only $0.1\%\text{--}3\%$ of parameters.

## 3. Training Reparameterization & Zero Runtime Cost
- **Training:** Parameterized via bottleneck MLP ($P = \operatorname{MLP}(E)$) to ensure smooth gradient landscapes during early training.
- **Inference:** The MLP is discarded; static materialized prefixes $P_K^{(l)}, P_V^{(l)}$ are prepended directly into the KV-cache, incurring zero parameter inference overhead.
