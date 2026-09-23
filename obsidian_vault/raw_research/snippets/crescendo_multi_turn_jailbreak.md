# Crescendo: Multi-Turn Cognitive Drift (Russinovich et al., USENIX Security 2025)

## 1. Single-Turn Moderation Vulnerability
Safety filters and guardrails primarily evaluate single-turn user prompts for malicious keywords or toxic intent vectors. Mark Russinovich, Ahmed Salem, and Ronen Eldan (*Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack*, USENIX Security 2025 / arXiv:2404.01833) introduce **Crescendo**, demonstrating that safety alignment can be bypassed by gradually entraining the model's self-generated context over multi-turn dialogues.

## 2. Attack Dynamics & Mechanistic Refusal Suppression
1. **Benign Anchoring:** Initiates conversation with an innocuous, historically or academically framed question that the model safely answers.
2. **Recursive Self-Referencing:** The adversary asks follow-up questions explicitly referencing the model's *own previously emitted tokens* (`"Great, now expand on component X from your previous answer..."`).
3. **Refusal Subspace Inactivation:**
   Because the user prompt contains no forbidden keywords and the model attends heavily to its own benign outputs in the KV cache, the 1D refusal direction $\hat{\mathbf{r}}$ remains dormant.
4. **Cognitive Drift:** Over $4\text{--}10$ dialogue turns, self-attention steers activations progressively into dangerous capability manifolds, yielding restricted outputs.

## 3. Defense Frontier
- **Multi-Turn Trajectory Moderation:** Auditing cumulative conversational drift across the entire dialogue history.
- **[[representation_circuit_breakers_zou|Representation Circuit Breakers]]:** Halting generation whenever internal latent manifolds enter prohibited subspaces, regardless of conversational turn structure.
