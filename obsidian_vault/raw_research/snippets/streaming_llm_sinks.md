# StreamingLLM: Attention Sinks & Register Tokens (Xiao et al. ICLR 2024)

## 1. The Softmax Normalization Dilemma
Self-attention enforces unit-sum normalization $\sum_{j=1}^t A_{t, j} = 1$. When intermediate tokens do not require strong conditioning on preceding context, the query vector $q_t$ still must allocate probability mass across prior tokens.
- **Attention Sinks:** Because initial tokens ($[0:4]$) are visible to every subsequent token across causal attention masks, early layers spontaneously repurpose them as numerical anchors to dump excess probability mass.
- **Sliding-Window Perplexity Explosion:** Evicting initial tokens in naive sliding-window caches destroys the Softmax denominator normalization, triggering immediate catastrophic perplexity explosion ($\text{PPL} \to 10^4+$) after step $W+1$.

## 2. StreamingLLM Architecture (Xiao et al. ICLR 2024)
Decouples local context retention from attention normalization:
$$\text{Cache}_t = \underbrace{\{x_0, x_1, x_2, x_3\}}_{\text{Initial Attention Sinks (Fixed)}} \cup \underbrace{\{x_{t-W+4}, \dots, x_t\}}_{\text{Rolling Context Window}}$$
- **Positional Re-centering:** Under Rotary Position Embeddings (RoPE), rolling-window tokens are assigned relative cache coordinates $[4, W]$ rather than absolute document steps $t$, matching pre-trained distance metrics.
- **Empirical Longevity:** Sustains stable, non-exploding perplexity across **4,000,000+ tokens** of continuous streaming text without requiring any model fine-tuning or weight modification.

## 3. Cross-Modal Parallel: Register Tokens in ViTs (Darcet et al. ICLR 2024)
Non-causal Vision Transformers lack a fixed initial token, leading models to repurpose low-information background patches as computational registers, creating severe high-norm artifacts. Prepending $4\text{--}8$ explicit learnable register tokens $[REG]$ absorbs excess attention mass, removing feature artifacts and purifying attention heatmaps.