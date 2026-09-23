# Frontier Research Pivot Directives (Phase VII - 2026-09-23)

## 1. Executive Summary & Directional Scope
Phase VII advances frontiers in real-time acoustic steerability, non-causal grammar diffusion, and statistical manifold uncertainty quantification.

---

## 2. Pillar 1: Full-Duplex Speech-Language Co-Reasoning (Kyutai Mimi / Moshi Architecture)
- **Problem Formulation:** Conventional conversational voice systems operate via cascaded pipelines: Automatic Speech Recognition (ASR) $\to$ LLM reasoning $\to$ Text-to-Speech (TTS). This introduces a minimum latency bottleneck of $800\text{--}1500\,\text{ms}$, completely destroying conversational naturalness, interruption dynamics, and acoustic emotional nuance.
- **Proposed Frontier Formulation:** A speech-native multimodal foundation architecture (Kyutai Mimi / Moshi). Uses a neural audio codec operating at $12.5\,\text{Hz}$ with Residual Vector Quantization (RVQ, 8 codebooks per frame), interleaving acoustic and textual tokens across concurrent causal audio streams. Latency is compressed to $<160\,\text{ms}$, enabling true full-duplex conversational interruption and direct prosodic activation steering via internal hidden layer hooks.
- **Actionable Directive:** Formulate a mathematical directive for direct activation addition on prosody and emotion steering vectors within speech-native transformer layers.

---

## 3. Pillar 3: Non-Causal Grammar-Constrained Discrete Diffusion (SEDD-cFSM)
- **Problem Formulation:** Existing grammar-constrained decoding engines (XGrammar, LLGuidance) are strictly designed for left-to-right causal autoregressive generation. Discrete diffusion models (SEDD, Plaid, MDLM) generate text via continuous-time Markov jump processes with bidirectional infilling, rendering standard causal token bitmasks mathematically invalid.
- **Proposed Frontier Formulation:** Formulate a non-causal pushdown automaton constraint engine for discrete score-based diffusion. Compute bidirectional prefix-suffix intersection masks over partial sequences:
  $$\mathcal{M}_{\text{diff}}(t, i) = \left\{ w \in \Sigma \mid \exists \text{ path in } \mathcal{G} \text{ compatible with } x_{-i} \text{ and } x_{+i} \right\}$$
  Directly mask reverse-time jump rates during tau-leaping SDE integration, guaranteeing that completed non-autoregressive infillings satisfy formal programming grammars.
- **Actionable Directive:** Benchmark grammar-constrained discrete diffusion on structured code generation and JSON synthesis.

---

## 4. Pillar 3: Manifold Epistemic Calibration via Fisher Metric Volume Elements
- **Problem Formulation:** Semantic Entropy measures epistemic uncertainty through discrete clustering across $N=10$ stochastic samples. However, discrete clustering ignores the continuous geometry of the probability simplex and intermediate activation manifolds.
- **Proposed Frontier Formulation:** Quantify epistemic uncertainty by calculating the Riemannian metric tensor $g_{ij} = \mathbb{E}[\partial_i \log p \cdot \partial_j \log p]$ (Fisher Information Matrix) and evaluating the local differential volume element:
  $$dV = \sqrt{\det(G)} \, dx_1 \wedge \dots \wedge dx_d$$
  Regions where the volume element collapses or expands abnormally indicate high epistemic instability and hallucination propensity, enabling calibrated single-pass uncertainty scoring.
- **Actionable Directive:** Integrate Fisher volume calibration into test-time verifier loops.
