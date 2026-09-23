# Strategic Research Pivot 11: Multi-Token Planning, RoPE Frequency Partitioning & Game-Theoretic Self-Play Alignment

## 1. Executive Summary & Pivot Directive
Expanding upon Double Early Exiting (Kangaroo), Length-Normalized Alignment (SimPO), and Prompt-Adaptive Compute Budgets (Snell et al.), Strategic Pivot 11 introduces three critical frontiers:
1. Transforming autoregressive next-token prediction into parallel planning heads with dual-token speculative generation (DeepSeek-V3 MTP).
2. Overcoming context-window extrapolation boundaries via frequency-band partitioned Fourier interpolation and attention entropy calibration (YaRN).
3. Resolving the intransitivity failure mode of scalar reward modeling through game-theoretic two-player zero-sum Nash equilibrium alignment (SPPO).

---

## 2. Three Novel Research Horizons

### Horizon 1: Multi-Token Prediction (MTP) Speculative Architecture & Planning Horizons (DeepSeek-V3 / Gloeckle et al., Meta 2024)
- **Problem Statement:** Standard next-token cross-entropy forces representation learning to remain greedy, depriving intermediate layers of multi-token lookahead planning signals and imposing a 1-token-per-pass inference latency barrier.
- **Frontier Architecture:** *Multi-Token Prediction (MTP)*. Cascades auxiliary prediction modules with shared unembedding projections to predict future tokens $x_{i+1:i+D}$ simultaneously. During inference, these prediction heads serve as zero-overhead speculative draft generators, accelerating decoding by **$1.8\times$** without external draft models.
- **Focus Area:** Sequential representation fusion $[h_i; \text{Emb}(x_{i+k})]$, shared unembedding parameter tying, and dual-token parallel verification in serving engines.

### Horizon 2: Frequency-Band Partitioned RoPE Scaling & Attention Entropy Calibration (YaRN, Peng et al., ICLR 2024)
- **Problem Statement:** Naive Position Interpolation (PI) compresses all rotation frequencies uniformly, destroying high-frequency local grammar and punctuation awareness. Conversely, context scaling dilutes softmax attention sharpness across long contexts.
- **Frontier Architecture:** *YaRN (ICLR 2024 / arXiv:2309.00071)*. Partitions RoPE dimensions into three distinct wavelength regimes ($\lambda_i$ relative to context $L$): high frequencies remain un-interpolated, low frequencies are linearly interpolated, and mid frequencies blend smoothly via a ramp function $\gamma(r_i)$. Incorporates a calibrated temperature scaling factor $\sqrt{t} = \sqrt{0.1 \ln(s) + 1}$ to preserve attention entropy.
- **Focus Area:** 128k context extrapolation with $10\times$ fewer training tokens, zero degradation on short-context retrieval, and needle-in-a-haystack verification.

### Horizon 3: Game-Theoretic Self-Play Preference Optimization (SPPO, Wu et al., ICML 2024)
- **Problem Statement:** Classical alignment frameworks (RLHF, DPO, SimPO) assume preferences obey the Bradley-Terry model. However, human and model preferences systematically exhibit **intransitivity and Condorcet cycles** ($A \succ B \succ C \succ A$). Fitting a scalar reward to cyclic data induces reward hacking and optimization collapse.
- **Frontier Architecture:** *Self-Play Preference Optimization (SPPO, ICML 2024 / arXiv:2405.00675)*. Reframes alignment as finding the **Nash Equilibrium of a two-player symmetric zero-sum game**. Solves for the unexploitable policy $\pi^*$ iteratively via self-play candidate sampling and Multiplicative Weights Updates (MWU).
- **Focus Area:** Intransitive preference handling, unexploitable policy convergence, and self-play alignment without expensive proprietary model distillation.

---

## 3. Directives for Immediate Research Execution
1. Codify **Section 60** (DeepSeek-V3 MTP Speculative Architecture), **Section 61** (YaRN Context Extension & Attention Calibration), and **Section 62** (SPPO Game-Theoretic Alignment) in `latent_mechanics_dossier.md`.
2. Author atomic notes in `obsidian_vault/raw_research/snippets/`:
   - `mtp_multi_token_prediction_architecture.md`
   - `yarn_rope_frequency_scaling.md`
   - `sppo_self_play_nash_alignment.md`
3. Rebuild `000_Master_Brain_Index.md` using `scripts/build_obsidian_index.py`.
4. Commit and push persistently to GitHub `origin/main`.
