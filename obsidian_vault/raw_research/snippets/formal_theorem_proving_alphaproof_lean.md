# Formal Theorem Proving with LLMs: AlphaProof & Lean-STaR (2024)

## Bridging Intuition & Formal Verification
Standard Chain-of-Thought (CoT) suffers from subtle logical leaps and hallucinations in advanced mathematics. Neurosymbolic theorem proving couples LLM search with interactive theorem provers (ITPs) like **Lean 4**, where proofs are deterministically checked by a proof kernel.

## 1. AlphaProof (Google DeepMind, 2024)
Silver-medal standard performance on the International Mathematical Olympiad (IMO):
- **Architecture:** Combines pre-trained language models (Gemini) with AlphaZero-style Reinforcement Learning (RL) and Monte Carlo Tree Search (MCTS).
- **Lean Formalization:** Natural language problem statements are automatically auto-formalized into Lean 4 specifications.
- **Test-Time Search:** Allocates extensive test-time compute to explore proof search trees in Lean. The Lean compiler serves as an infallible ground-truth verifier, rewarding valid tactic applications and pruning dead ends.

## 2. Lean-STaR: Interleaving Thinking & Proving (2024)
Addresses the gap between informal reasoning and formal tactic execution:
- **Interleaved Thought Generation:** Rather than directly predicting Lean tactics ($t$), prompts the model to emit natural language "informal thoughts" ($\tau$) outlining strategic intent before generating formal tactics: $(x \to \tau \to t)$.
- **Self-Taught Reasoner (STaR):** Uses successful Lean-verified proofs to generate synthetic intermediate thoughts retrospectively, fine-tuning the model to align informal scratchpads with formal verification rules.
- **Results:** Substantially outperforms direct tactic generation on `miniF2F`, proving that explicit scratchpad reasoning guides formal state transitions.
