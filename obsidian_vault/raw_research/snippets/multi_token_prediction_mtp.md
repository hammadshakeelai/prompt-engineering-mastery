# Multi-Token Prediction (MTP) & Native Self-Speculation (Gloeckle et al., ICML 2024; DeepSeek-V3)

## 1. Limitations of Next-Token Prediction (NTP)
Standard autoregressive language models minimize the cross-entropy loss over a single forward step:
$$\mathcal{L}_{\text{NTP}} = -\sum_{t=1}^T \log P(x_t \mid x_{<t})$$
- **Local Myopia:** NTP incentivizes the model to fit high-frequency local syntax rather than planning long-range algorithmic solutions.
- **Inference Inefficiency:** Sampling remains strictly sequential: generating $N$ tokens requires $N$ sequential network forward passes.

## 2. Multi-Token Architecture & Objective
Gloeckle et al. (Meta FAIR, ICML 2024) introduce Multi-Token Prediction (MTP), predicting $M$ future tokens simultaneously:
$$\mathcal{L}_{\text{MTP}} = -\frac{1}{M} \sum_{k=1}^M \sum_{t=1}^T \log P_k(x_{t+k} \mid x_{\le t})$$
- **Sequential MTP Blocks (DeepSeek-V3 Formulation):**
  A shared trunk outputs hidden state $h_t^{(0)}$. Each subsequent token $k \in [1, M]$ is predicted via a lightweight transformer block that fuses the prior hidden representation with the embedding of the preceding predicted token:
  $$h_t^{(k)} = \text{TransformerBlock}_k\left( \left[ h_t^{(k-1)} \,\|\, \text{Emb}(x_{t+k-1}) \right] \right)$$
  $$P_k(x_{t+k} \mid x_{\le t}) = \text{Softmax}\left( W_{\text{unembed}} h_t^{(k)} \right)$$
- **Gradient Isolation:** In DeepSeek-V3, MTP heads share the unembedding matrix $W_{\text{unembed}}$ with the main model trunk, stabilizing training dynamics without memory blowup.

## 3. Native Self-Speculative Decoding
- **Zero Draft Model Overhead:** Rather than maintaining a separate smaller draft model (and dual KV caches), the primary model's own MTP heads draft $M$ future tokens in parallel during a single forward pass.
- **Verification Loop:** At the next step, the main trunk verifies the drafted tokens using standard speculative acceptance criteria:
  $$\alpha_k = \min\left(1, \frac{P_{\text{main}}(x_{t+k} \mid x_{<t+k})}{P_{\text{MTP}}(x_{t+k} \mid x_{\le t})}\right)$$
- **Performance Gains:** Boosts code generation pass rates ($+12\%$ HumanEval, $+17\%$ MBPP) and expands inference throughput by $1.8\times\text{--}3.0\times$.
