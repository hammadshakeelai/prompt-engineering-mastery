---
name: speculative-decoding-specialist
description: Specialized directive for speculative decoding algorithms, draft model topologies, Medusa/Eagle recurrent feature-level drafting, acceptance rate optimization (LK losses), tree-attention verification, and grammar-synchronized draft logit masking.
---

# Speculative Decoding Specialist Skill

Use this skill when designing high-throughput LLM inference engines, optimizing draft-target model pairings, engineering tree-attention verification kernels, or tuning speculative acceptance rates.

## 1. Mathematical Foundations & Acceptance Mechanics

1. **Exact Distribution Preservation (Leviathan et al. / Chen et al.):**
   - Let $P_t(x)$ denote target model probability and $P_d(x)$ denote draft model probability.
   - Accept draft token $x$ with probability:
     $$\alpha(x) = \min\left(1, \frac{P_t(x)}{P_d(x)}\right)$$
   - If rejected, sample replacement token from the normalized residual distribution:
     $$P_{\text{res}}(x) = \frac{\max(0, P_t(x) - P_d(x))}{\sum_{x'} \max(0, P_t(x') - P_d(x'))}$$
   - Guarantees that the output distribution matches the target model exactly: $\mathbb{E}[P_{\text{out}}] \equiv P_t$.

2. **Total Variation Bound & Acceptance Expectation:**
   - Expected token acceptance rate relates directly to Total Variation distance:
     $$\mathbb{E}[\alpha] = 1 - \frac{1}{2} \|P_t - P_d\|_{\text{TV}}$$
   - Target an average acceptance length $\tau_{\text{accept}} \ge 3.0$ tokens per target model forward step to overcome verification kernel invocation latency.

## 2. Drafting Topologies & Architectures

1. **Independent Autoregressive Draft Models (SpD):**
   - Pair target model (e.g., 70B) with compact draft model sharing the identical tokenizer (e.g., 1B–3B).
   - Align Key-Value projections using cross-model transfer adapters where feasible to avoid duplicate prefill memory footprint.

2. **Residual Head Drafting (Medusa):**
   - Append multiple parallel MLP prediction heads to top-layer target hidden states, predicting $k$ future tokens concurrently without maintaining a separate draft model.
   - Combine with tree-attention masks to verify candidate combinations in a single forward pass.

3. **Recurrent Feature Drafting (EAGLE & EAGLE-2):**
   - Draft future token sequences in the **feature space** rather than raw token embedding space by passing top hidden states through a single lightweight transformer decoder layer.
   - Employ contextual tree pruning to dynamically adjust tree depth based on confidence entropy, achieving $2.5\times\text{--}3.5\times$ speedups.

## 3. Tree-Attention Batched Verification

1. **Tree Mask Construction:**
   - Structure draft proposals as a non-linear tree of branching token hypotheses rather than linear chains.
   - Construct custom 2D causal tree attention masks $M_{\text{tree}} \in \{0, 1\}^{B \times B}$ where $M_{i,j} = 1$ if node $j$ is an ancestor of node $i$.
   - Execute verification in a single forward pass of the target model, maximizing GPU Tensor Core utilization.

## 4. Grammar-Synchronized Speculative Decoding

1. **Draft-Stage Syntax Masking:**
   - Apply Context-Free Grammar (CFG) or DFA bitmasks directly to draft model logits prior to sampling, preventing the draft model from proposing illegal syntax paths that would be rejected deterministically by formal validators.

2. **Zero-Compute Deterministic Jumping:**
   - When the parsing state dictates a deterministic next-token sequence (e.g., fixed JSON keys, keyword syntax), bypass draft model inference entirely and inject the deterministic token slice directly into the target verification buffer.

## 5. Loss Function Alignment (LK Losses)

1. **Avoid Forward KL Mode-Covering:**
   - Standard forward KL divergence $\mathcal{D}_{\text{KL}}(P_t \,\|\, P_d)$ forces the draft model to cover all target modes, smearing probability mass and depressing top-1 acceptance.
2. **Acceptance-Targeted Objective:**
   - Train draft models using direct acceptance or LK loss objectives (vLLM / Samarin et al.):
     $$\mathcal{L}_{\text{accept}} = -\mathbb{E}_{x \sim P_d} \left[ \min\left(1, \frac{P_t(x)}{P_d(x)}\right) \right]$$
   - Yields $+8\%\text{--}+12\%$ higher empirical acceptance rates with zero inference overhead.
