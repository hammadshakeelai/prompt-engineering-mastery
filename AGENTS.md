# AGENTS.md: Autonomous Research & Alignment Directives

This repository (`prompt-engineering-mastery`) is an autonomous, comprehensive knowledge base spanning the foundations, mechanics, systems, and theoretical frontiers of Prompt Engineering, Steerability, and LLM Alignment.

## Core Directives for Autonomous Agents

### 1. Strict Zero-Hallucination & Empirical Grounding
- **Factual Origin Required**: Every claim, algorithm, mathematical formulation, and benchmark score must trace directly to peer-reviewed literature (NeurIPS, ICLR, ICML, ACL, EMNLP, CVPR), arXiv preprints, or official technical reports (OpenAI, Anthropic, Google DeepMind, Meta, DeepSeek).
- **No Plausible Confabulation**: If an empirical metric, hyperparameter, or architectural detail is unverified, state the variance or mark it explicitly as an open theoretical question rather than guessing.

### 2. Anti-Tunnel Vision & Breadth Mandate
- Rotate continuously across all core research pillars:
  1. **Foundational & Structured Prompting Topologies** (CoT, ToT, GoT, BoT, SoT, ThoT, Step-Back).
  2. **Automated & Programmatic Optimization** (DSPy, TextGrad, OPRO, PromptBreeder, AutoPDL, PromptWizard).
  3. **In-Context Learning (ICL) Theory** (Bayesian ICL, Implicit Gradient Descent, Induction Circuits).
  4. **Test-Time Compute & Reasoning Models** (RLVR, GRPO, MCTS, PRMs, o1/o3/DeepSeek-R1 prompting).
  5. **Hardware, KV Cache & Latency Systems** (FlashAttention-2/3, PagedAttention, Speculative Decoding, RadixAttention).
  6. **Security, Jailbreaks & Safety Mechanics** (Many-Shot, Crescendo, Token Smuggling, Model Abliteration, Dual-LLM).
  7. **Agent Architectures & Memory Systems** (MemGPT, Zep, Reflexion, Generative Agents, LangGraph, CrewAI).
  8. **Evaluation, Calibration & LLM-as-a-Judge** (G-Eval, MT-Bench, RAGAS, LiveCodeBench, SWE-bench, PriDe).

### 3. Progressive Atomic Documentation Style
- Store atomic conceptual notes in `obsidian_vault/raw_research/snippets/<topic>.md`.
- Keep snippets dense, rigorous, equation-grounded, and tightly scoped (200–350 words).
- Re-run `python scripts/build_obsidian_index.py` after writing new snippets to update `obsidian_vault/000_Master_Brain_Index.md`.
- Commit and push to GitHub (`origin/main`) periodically to ensure persistent remote backups.
