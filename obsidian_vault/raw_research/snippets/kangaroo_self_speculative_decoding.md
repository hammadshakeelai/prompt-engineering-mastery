# Kangaroo Lossless Self-Speculative Decoding (Liu et al., NeurIPS 2024)

## 1. Architectural Motivation & Limitations of Independent Draft Models
Speculative decoding typically pairs a large target LLM with an independent smaller draft model (e.g., Llama-68M drafting for Llama-70B). This introduces memory fragmentation, independent KV cache allocations, and tokenizer/vocabulary alignment overhead. Previous self-speculative approaches suffered from low draft acceptance rates or rigid draft budgets.

## 2. Double Early Exiting Mechanics
Fangcheng Liu et al. (*Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting*, NeurIPS 2024 / arXiv:2404.18911) propose a double early exit framework:

1. **Sub-Network Early Exiting (Drafting):**
   - The draft model is derived directly from the first $L_s$ shallow layers of the frozen target LLM ($L_s \ll L$, e.g., layers $1\dots 4$ of a 32-layer LLM).
   - A lightweight adapter module $A_\phi$ (typically a single Transformer decoder layer, $\approx 1\%$ total parameter overhead) bridges the representation gap between shallow representations $h^{(L_s)}$ and final target logits:
     $$\hat{P}_{\text{draft}}(x_{t+1} \mid x_{\le t}) = \operatorname{Softmax}\left( W_{\text{unembed}} A_\phi(h_t^{(L_s)}) \right)$$
2. **Dynamic Early Exiting (Confidence-Based Halting):**
   - Drafting beyond the model's confidence threshold wastes compute on tokens that will be rejected by the target model.
   - At each draft step $\tau \in \{1, \dots, \gamma\}$, evaluate top-1 draft prediction confidence:
     $$c_\tau = \max_{v \in \mathcal{V}} \hat{P}_{\text{draft}}(v \mid x_{\le t+\tau-1})$$
   - If $c_\tau < \eta$ (calibrated threshold $\eta \in [0.6, 0.85]$), speculative drafting halts immediately.

## 3. Lossless Verification & Performance
- Target model evaluates drafted tokens in a single parallel verification pass via standard speculative rejection sampling:
  $$\alpha_i = \min\left(1, \frac{P_{\text{target}}(\tilde{x}_i \mid x_{<i})}{\hat{P}_{\text{draft}}(\tilde{x}_i \mid x_{<i})}\right)$$
  resampling from the normalized positive residual distribution upon rejection, guaranteeing exact output distribution equivalence ($P_{\text{Kangaroo}} \equiv P_{\text{target}}$).
- **Benchmark Speedup:** Delivers up to **$2.04\times$ wall-clock speedup** across Spec-Bench tasks (Chat, Reasoning, Code) while eliminating secondary model hosting overhead.
