# Glitch Tokens & Embedding Centroid Geometry (Rumbelow & Watkins 2023)

## 1. The Tokenizer-Training Data Mismatch
Modern BPE tokenizers are trained on raw web corpora containing frequent Reddit usernames (`" SolidGoldMagikarp"`, `" TheNitromeFan"`), bot handles, and forum formatting markers.
- **Pre-Training Filtering:** When pre-training datasets are cleaned and deduplicated, these specific strings are removed.
- **Zero Gradient Signal:** Over hundreds of billions of training tokens, their embedding vectors $W_E[i]$ and unembedding vectors $W_U[i]$ receive near-zero gradients:
  $$\nabla_{W_E[i]} \mathcal{L} \approx 0 \quad \text{and} \quad \nabla_{W_U[i]} \mathcal{L} \approx 0$$

## 2. Geometric Pathology in Latent Space
Under AdamW weight decay $\frac{\lambda}{2} \|W\|_2^2$, under-trained embeddings drift toward the origin or cluster tightly around the geometric centroid:
$$W_E[i] \approx \bar{W}_E = \frac{1}{|V|} \sum_{v \in V} W_E[v]$$
- **Centroid Proximity:** Proximity to the centroid means the token vector has near-constant inner product with arbitrary hidden states: $\langle W_E[i], h \rangle \approx c$.
- **Norm Anomalies:** Euclidean norms $\|W_E[i]\|_2$ deviate wildly from standard semantic token norms.

## 3. Activation Collapse & Behavioral Manifestation
When prompted to process or repeat a glitch token:
1. **Attention Disorientation:** Self-attention heads produce uniform, entropy-maximized attention distributions, failing to form induction heads.
2. **Latent Space Amplification:** Subsequent SwiGLU / MLP blocks project hidden states into unconstrained out-of-distribution regions.
3. **Unembedding Distortion:** Logits at the output layer collapse into repetitive token loops (repeating `"distribute"`, evasive strings) or hallucinated emotional refusals.
