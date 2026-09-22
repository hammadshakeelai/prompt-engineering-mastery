# Strategic Research Pivot 05: Long-Term Neural Memory, Formal Verification & Audio Latents

## 1. Executive Summary & Pivot Directive
Having mapped the frontiers of constrained decoding (llguidance, cFSMs, XGrammar), submodular KV eviction (H2O, SnapKV), and steerability (CAA, Crosscoders), this pivot establishes three transformative research horizons for 2025–2026:

---

## 2. Three Novel Research Horizons

### Horizon 1: Test-Time Associative Memory & Hybrid Transformers (Google Titans & SSD)
- **Problem Statement:** Standard quadratic attention ($O(T^2)$) and linear recurrent models (Mamba, RWKV) force a trade-off between exact recall and constant memory.
- **Frontier Architecture:** *Titans: Learning to Memorize at Test Time* (Google Research, Jan 2025). Integrates sliding-window attention (short-term) with a Neural Long-Term Memory (NLTM) module updated via test-time gradient steps governed by a surprise metric (momentary vs. past surprise) and adaptive forgetting gates.
- **Focus Area:** Scaling agent context beyond 2M+ tokens with $O(1)$ memory footprint and constant latency.

### Horizon 2: Formal Theorem Proving & Dual-Loop Reinforcement Learning (Lean 4 & AlphaProof)
- **Problem Statement:** Natural language chain-of-thought suffers from ungrounded informal leaps and uncheckable hallucinations in mathematical and algorithmic proofs.
- **Frontier Architecture:** Neuro-symbolic formal verification (DeepSeek-Prover-V1.5, AlphaProof, LeanCopilot). A dual-loop reasoning topology where an informal policy model generates tactic drafts while an interactive theorem prover (Lean 4 REPL) provides deterministic kernel verification feedback.
- **Focus Area:** Ground-truth Process Reward Models (PRMs) trained exclusively on verifiable Lean proof states.

### Horizon 3: Speech-Native Audio Conditioning & Latent Steerability (Moshi, Gemini 2.0 Audio)
- **Problem Statement:** Cascaded voice pipelines (ASR $\to$ LLM $\to$ TTS) destroy non-verbal prosody, emotional micro-inflections, and introduce 500ms+ acoustic latency.
- **Frontier Architecture:** Speech-native autoregressive models operating directly on Residual Vector Quantization (RVQ) discrete acoustic tokens (SoundStream, SNAC).
- **Focus Area:** Steerability of acoustic latent streams (pacing, vocal emotion, natural conversational turn-taking, interruption handling) via activation patching and semantic-acoustic cross-attention.

---

## 3. Directives for Next Subagent Iterations
1. Deep-dive on Google's **Titans** architecture, formulating test-time surprise updates and the MAG/MAC/MAL memory variants.
2. Formalize **Glitch Tokens and Embedding Geometry**, analyzing why under-trained tokens cluster near the centroid of representation space and destabilize decoding.
3. Synchronize new conceptual notes into the Obsidian knowledge graph and maintain remote backup.
