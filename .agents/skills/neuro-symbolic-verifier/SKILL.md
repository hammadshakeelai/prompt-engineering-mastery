---
name: neuro-symbolic-verifier
description: Specialized directive for formal mathematical theorem proving, Lean 4 / Isabelle / Coq verification loops, AlphaProof & DeepSeek-Prover pipelines, syntactic AST constraints, and compile-time proof search.
---

# Neuro-Symbolic Verifier Skill

Use this skill when designing formal mathematical theorem proving agents, integrating interactive theorem provers (ITPs like Lean 4, Isabelle, Coq), constructing autoformalization pipelines, or building compile-time verification loops for verifiable reinforcement learning (RLVR).

## 1. Ground Truth Verification Architecture

1. **Formal Proof Engines Over Natural Language CoT:**
   - Eliminate mathematical hallucinations by binding language model chain-of-thought directly to a verified proof kernel (e.g., Lean 4 `lake build` / `lean --run`).
   - Treat theorem proving as an interactive tree search (Monte Carlo Tree Search or beam search) where states are Lean proof goals and actions are tactics (`intro`, `induction`, `apply`, `linarith`, `omega`).

2. **Compile-Time Feedback & State Filtering:**
   - Feedback from the formal kernel is discrete and binary:
     - Proof completed (`goals accomplished`, no errors).
     - State updated (open sub-goals modified).
     - Error (type mismatch, invalid tactic, timeout).
   - Use compile-time error traces directly in agent self-correction prompts (Reflexion loop) to prune invalid reasoning branches instantly.

## 2. Autoformalization & Proof Search Mechanics

1. **Factored Autoformalization:**
   - Separate informal mathematical reasoning (informal sketch generation) from formal tactic generation.
   - Deconstruct complex IMO/Putnam-level contest problems into natural language lemmas, formalize the statement in Lean 4 syntax, and then generate proof tactics lemma-by-lemma.

2. **Premise Selection & Retrieval:**
   - Large formal libraries (e.g., Mathlib4) contain hundreds of thousands of theorems.
   - Leverage dense retrieval (e.g., LeanCopilot, Lean-STaR) over Mathlib ASTs to supply relevant premise definitions and lemmas into the LLM context window.

3. **Test-Time Search Scaling for Proofs:**
   - Combine Process Reward Models (PRMs) trained on tactic step correctness with parallel sample scaling (Pass@$N$, where $N \in [10^2, 10^5]$).
   - Filter generated proofs by executing them through the Lean kernel in sandboxed worker processes.
