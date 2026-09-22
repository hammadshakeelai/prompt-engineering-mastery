# Contrastive Activation Addition & Linear Representations (Rimsky 2023, Marks & Tegmark 2023)

## 1. The Linear Representation Hypothesis
High-level semantic traits (truthfulness, sycophancy, refusal, toxicity) are represented as one-dimensional linear directions $\hat{d} \in \mathbb{R}^D$ within the intermediate residual streams ($L/2 \text{ to } 3L/4$) of transformer models:
$$\text{TraitSalience}(h) = \hat{d}^T h$$
Marks & Tegmark demonstrated that truth directions discovered on simple geography or arithmetic statements generalize across diverse factual domains.

## 2. Contrastive Activation Addition (CAA)
Rimsky et al. formalized **CAA** to extract and inject steering vectors at runtime:
1. **Extraction:** From paired contrastive prompts differing by a target trait (e.g., agreeing with user misconceptions vs. asserting factual truth):
   $$v_{\text{steer}}^{(l)} = \frac{1}{|D|} \sum_{(x^+, x^-) \in D} \left( h^{(l)}(x^+) - h^{(l)}(x^-) \right)$$
2. **Intervention:** Perturb activations during decoding:
   $$h^{(l)} \leftarrow h^{(l)} + \alpha \cdot v_{\text{steer}}^{(l)}$$
   Steers model responses continuously without modifying model weights.

## 3. Orthogonal Concept Ablation
To permanently eliminate unwanted behaviors (e.g., sycophantic hallucinations) without dynamic tuning:
$$h_{\text{ablated}}^{(l)} = h^{(l)} - \left( h^{(l)} \cdot \hat{v} \right) \hat{v}$$
Projects internal activations onto the orthogonal null space of the undesirable direction, neutralizing the trait while retaining general reasoning capabilities.
