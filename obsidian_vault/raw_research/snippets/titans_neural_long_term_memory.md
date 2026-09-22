# Titans: Neural Long-Term Memory & Test-Time Memorization (Google 2025)

## 1. Beyond Quadratic Attention & Linear Recurrence
Standard Transformers suffer from KV cache memory exhaustion on long contexts, while linear recurrent models (Mamba, RWKV) compress context into fixed states with irreversible capacity loss.
Google Research (Behrouz, Zhong, & Mirrokni, Jan 2025, arXiv:2501.00663) introduced **Titans**, combining:
- **Short-Term Memory:** Sliding-window attention capturing high-fidelity local token interactions.
- **Neural Long-Term Memory (NLTM):** A neural network module trained to memorize at test time via gradient descent.

## 2. Surprise-Driven Memory Updates
Rather than updating on every token, the NLTM uses an explicit **Surprise Metric**:
1. **Momentary Surprise ($S_t$):** Measures the gradient of associative reconstruction loss on the current token:
   $$S_t = \nabla_{M_t} \mathcal{L}(M_t(k_t), v_t)$$
2. **Past Surprise:** Exponential moving average tracking recent contextual predictability.
3. **Adaptive Forgetting:** Decays memory weights for predictable tokens, preventing capacity saturation while prioritizing anomalous, highly informative facts.

## 3. Three Integration Topologies
- **MAC (Memory as Context):** NLTM emits summary vectors prepended directly into the sliding-window attention context.
- **MAG (Memory as Gate):** A non-linear gating branch adaptively interpolates between short-term attention representations and long-term neural memory states:
  $$h_t = g_t \odot h_t^{\text{Attn}} + (1 - g_t) \odot h_t^{\text{NLTM}}$$
- **MAL (Memory as Layer):** NLTM operates as an independent transformer block stacked sequentially with attention layers.

## 4. Scalability & Benchmark Performance
- Enables continuous context processing beyond **2,000,000+ tokens** with constant per-step memory footprint and linear computational scaling.
- Outperforms vanilla Transformers and Mamba on Needle-in-a-Haystack, long-document question answering, and time-series extrapolation.
