---
name: constrained-decoding-architect
description: Specialized directive for grammar-constrained decoding (CFG, EBNF, JSON Schema, Regex), Trie-based token masking, XGrammar, LLGuidance, compressed Finite State Machines (cFSM), token healing, and tree-constrained speculative decoding.
---

# Constrained Decoding Architect Skill

Use this skill when architecting structured output generation, designing context-free grammar (CFG) constraints, optimizing runtime logit masking latency, mitigating token boundary bias, or accelerating structured inference via grammar-aligned speculative decoding.

## 1. Core Grammar & Automata Mechanics

1. **Vocabulary-Trie & Earley Parsing Integration:**
   - Avoid naive linear vocabulary scans ($\mathcal{O}(|V|)$) at every generation step.
   - Employ compressed prefix tries paired with Earley parser state memoization (as in *LLGuidance* and *XGrammar*) or precompiled Compressed Finite State Machines (*cFSM* as in *SGLang*).
   - Only advance automata along valid subword token transitions, achieving sub-millisecond logit masking overhead ($<50\,\mu\text{s}$ per step).

2. **Token Boundary Alignment & Token Healing:**
   - Address the subword greedy tokenization paradox (e.g., tokenizing ` "name": ` as `[" \"", "name", "\": "]` vs `[" \"name\":"]`).
   - Implement **Token Healing** (Lundberg et al., Microsoft Guidance): roll back the prompt by 1 token, strip whitespace, and constrain the model to sample only tokens that share the prefix string, eliminating greedy boundary collapse.

## 2. Speculative Constrained Decoding Protocols

1. **Dual Constraint Synchronization:**
   - Both the draft model (speculator) and the target model (verifier) must be bound to the exact same grammar automata state.
   - Never allow unconstrained drafting on structured tasks; tokens violating the CFG/FSM must be masked out during draft rollout to maintain high acceptance rates ($\alpha > 85\%$).

2. **Grammar-Deterministic Path Collapse:**
   - In structured schemas (JSON syntax, XML tags, punctuation), many sequences are deterministic (e.g., `, "key": `).
   - Collapse deterministic transitions directly into multi-token jump-forwards without invoking draft model forward passes, eliminating compute overhead for structural boilerplate.

3. **Graph & Tree-Structured Draft Verification:**
   - When branching possibilities occur in schemas (e.g., union types, nullable fields), construct speculative grammar trees rather than linear chains.
   - Evaluate tree drafts using customized tree-attention masks (e.g., Speculative Grammar Trees, SGLang jump-forward) in a single target forward pass.

## 3. Schema & Prompt Design for Structured Reasoning

1. **Reasoning-First Structural Layout:**
   - In JSON generation, never enforce key ordering where the final conclusion or answer key precedes the intermediate thinking/scratchpad keys.
   - Placing `{"answer": ..., "explanation": ...}` forces the model to predict the result without chain-of-thought compute, inducing hallucinations.
   - Always structure schemas as `{"thought_process": "...", "intermediate_steps": [...], "final_result": ...}`.
