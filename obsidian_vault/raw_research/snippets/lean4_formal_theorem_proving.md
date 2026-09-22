# Neuro-Symbolic Theorem Proving & Lean 4 Verification (AlphaProof, DeepSeek-Prover-V1.5)

## 1. Natural Language Informal Proof Fallibility
Informal mathematical Chain-of-Thought (CoT) in standard LLMs frequently confabulates subtle arithmetic errors, skips non-trivial boundary cases, and hallucinates justifications. In contrast, interactive theorem provers (Lean 4) verify proofs via the **Calculus of Inductive Constructions (CIC)**: a proof is logically valid if and only if it type-checks into a verified proof term (`no goals`).

## 2. Dual-Loop Autoformalization Architecture
1. **Informal Planner (System 2):** High-capability LLM generates natural language proof sketches, proposing intermediate subgoals and lemmas.
2. **Tactic Policy Model:** Translates informal steps into executable Lean 4 tactic scripts (`intro`, `apply`, `linarith`, `omega`, `ring`, `simp`).
3. **Deterministic REPL Kernel:** Lean 4 compiler executes the tactic against the active goal state $\Gamma \vdash G$. Compiler diagnostics (`type mismatch`, `unsolved goals`) provide non-hallucinated feedback for recursive backtracking.

## 3. Search & Test-Time Verification Topologies
- **DeepSeek-Prover-V1.5 (2024):** Utilizes **RLPAF** (Reinforcement Learning from Proof Assistant Feedback) and **RMaxTS** (Monte Carlo Tree Search with intrinsic exploration bonuses) to discover non-obvious tactic sequences, achieving 63.5% on miniF2F.
- **AlphaProof (DeepMind 2024):** Coupled Gemini formalization with an AlphaZero-style search engine. Utilized test-time RL across millions of synthetic formal problems, achieving silver-medal performance at the International Mathematical Olympiad (IMO).
- **Verifiable Process Reward Models (PRMs):** Lean 4 replaces fallible human/LLM process judges with deterministic proof-tree verification, providing absolute ground-truth step supervision.
