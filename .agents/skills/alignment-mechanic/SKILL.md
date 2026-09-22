---
name: alignment-mechanic
description: Specialized directive for mechanistic model steering, representation engineering (RepE), activation addition (CAA), sparse autoencoder (SAE) interventions, Crosscoder model diffing, and RLVR/GRPO reward alignment.
---

# Alignment Mechanic Skill

Use this skill when auditing, intervening upon, or designing steerability pipelines that operate on internal model representations, residual stream activations, or reinforcement learning feedback loops.

## 1. Representation & Activation Steering Directives

1. **Contrastive Activation Addition (CAA) & Steering Vectors:**
   - Extract steering vectors $v_{\text{steer}} = \mathbb{E}[x_{\text{positive}}] - \mathbb{E}[x_{\text{negative}}]$ across intermediate residual stream layers (typically layers $L/2$ to $3L/4$).
   - Apply runtime intervention: $x^{(l)} \leftarrow x^{(l)} + \alpha \cdot v_{\text{steer}}$, tuning coefficient $\alpha$ to maximize behavioral compliance without degrading perplexity or fluency.

2. **Representation Engineering (RepE):**
   - Use PCA or linear discriminant analysis on contrastive pair activations to identify reading vectors and control vectors for truthfulness, sycophancy, refusal, and epistemic caution.
   - Employ representation ablation (orthogonal projection) to neutralize unwanted biases: $x_{\text{ablated}} = x - (x \cdot \hat{v}) \hat{v}$.

3. **Sparse Autoencoders (SAEs) & JumpReLU:**
   - Decompose superimposed polysemantic activations into monosemantic latent directions: $f = \text{JumpReLU}_\theta(W_{\text{enc}} x + b_{\text{enc}})$.
   - Perform feature steering by directly clamping or boosting identified interpretable latent features before decoding back into residual space: $\hat{x} = W_{\text{dec}} f_{\text{steered}} + b_{\text{dec}}$.

## 2. Crosscoder Model Diffing Directives

1. **Model Differential Analysis:**
   - Use multi-layer, multi-model Crosscoders to identify features that are unique to aligned/instruct models versus their base predecessors.
   - Isolate refusal latents and sycophancy circuits that emerge exclusively from post-training RLHF/DPO.

2. **Cross-Layer Transcoder (CLT) Circuit Factorization:**
   - Linearize MLP computations into discrete input-to-output sparse features, mapping causal circuits through residual streams without layer-boundary artificial duplication.

## 3. RLVR & Policy Optimization Directives

1. **GRPO & Verifiable Rewards:**
   - Optimize policy models using group-normalized relative advantages: $A_i = \frac{R_i - \text{mean}(\{R\})}{\text{std}(\{R\})}$, discarding separate value network overhead.
   - Enforce strict ground-truth verifiers (unit tests, formal proofs, execution sandboxes) to prevent reward hacking and sycophancy.
