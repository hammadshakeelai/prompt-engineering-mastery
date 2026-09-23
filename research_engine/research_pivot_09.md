# Strategic Research Pivot 09: Process Supervision, Continuous Latent Reasoning & Vector-Indexed KV Memory

## 1. Executive Summary & Pivot Directive
Building upon Refusal Direction Geometry (Arditi et al.), Mooncake Disaggregated Serving (FAST 2025), EAGLE-2 Dynamic Tree Speculation, and Semantic Entropy Epistemic Uncertainty (Nature 2024), this pivot establishes three critical horizons bridging verifiable process reward feedback, implicit token-level reasoning, and sub-linear vector-quantized KV cache memory:

---

## 2. Three Novel Research Horizons

### Horizon 1: Automated Step-Level Process Supervision & Monte Carlo PRMs (Math-Shepherd, Wang et al., ACL 2024)
- **Problem Statement:** Outcome Reward Models (ORMs) provide sparse binary feedback ($R \in \{0, 1\}$) at the end of a long chain-of-thought, penalizing valid partial reasoning that failed at the final step, and rewarding false derivations that stumbled upon the correct answer (false positives).
- **Frontier Architecture:** *Math-Shepherd (ACL 2024 / arXiv:2312.08935)*. Automates step-level process supervision without human annotation. For each intermediate step $s_t$, perform $M$ parallel Monte Carlo rollouts to the terminal answer. The step reward $r(s_t)$ is the empirical proportion of rollouts terminating in the correct solution:
  $$r(s_t) = \frac{1}{M} \sum_{m=1}^M \mathbb{I}\left( \text{Rollout}_m(s_t) = y^* \right)$$
- **Focus Area:** Training step-level PRMs for test-time beam search, Best-of-$N$ candidate re-ranking, and step-level PPO policy optimization.

### Horizon 2: Continuous Token-Level Latent Reasoning & Pretrained Inner Monologues (Quiet-STaR, Zelikman et al., ICML 2024)
- **Problem Statement:** Chain-of-thought (CoT) reasoning is typically constrained to explicit prompt instructions or special `<think>` tags, remaining inaccessible during standard text pretraining and next-token prediction.
- **Frontier Architecture:** *Quiet-STaR (Stanford / ICML 2024 / arXiv:2403.09629)*. Language models learn to generate an "inner monologue" of latent thought tokens $t_1, \dots, t_T$ between raw sequence tokens $x_i$, trained via REINFORCE where the reward is the improvement in future token prediction likelihood:
  $$R_i = \sum_{j=1}^n \left( \log P(x_{i+j} \mid x_{\le i}, t_{1:T}) - \log P(x_{i+j} \mid x_{\le i}) \right)$$
- **Focus Area:** Parallel tokenwise thought generation, learned mixing heads ($\alpha_i$), and non-myopic future token prediction.

### Horizon 3: Product Quantization & Maximum Inner Product Search for Million-Token KV Memory (PQCache, Zhang et al., 2024)
- **Problem Statement:** Submodular eviction (H2O, SnapKV) permanently discards historical tokens, while linear scanning of long KV caches is memory-bandwidth bounded.
- **Frontier Architecture:** *PQCache (arXiv:2407.12820)*. Partitions Key vectors into $m$ orthogonal sub-vectors quantized into centroid codebooks $\mathcal{C}_1, \dots, \mathcal{C}_m$. During decoding, computes query-centroid lookup tables for asymmetric distance computation, executing Maximum Inner Product Search (MIPS) to fetch only top-$k$ attention blocks in $\mathcal{O}(1)$ time.
- **Focus Area:** Asymmetric MIPS tables, zero-overhead centroid caching, and sub-linear retrieval on 1M+ token contexts.

---

## 3. Directives for Immediate Research Execution
1. Append **Section 49** (Math-Shepherd Step Supervision), **Section 50** (Quiet-STaR Latent Monologues), and **Section 51** (PQCache Vector-Indexed KV Memory) to `latent_mechanics_dossier.md`.
2. Author atomic notes for all three topics in `obsidian_vault/raw_research/snippets/`.
3. Author deep-dive research document on **Automated Step Supervision & Monte Carlo Process Reward Models**.
4. Rebuild `000_Master_Brain_Index.md` and push persistently to GitHub `origin/main`.
