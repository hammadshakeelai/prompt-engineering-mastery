# Automated Circuit Discovery & Induction Heads (ACDC & EAP)

## 1. Induction Heads as the Mechanism of In-Context Learning
Olsson et al. (Anthropic, 2022) proved that In-Context Learning (ICL) is driven by a two-head induction circuit executing the pattern $[A][B] \dots [A] \to [B]$:
- **Previous-Token Head ($H_1$ in Layer $L$):** Encodes information about token $[A]$ into the representation at $[B]$.
- **Induction Head ($H_2$ in Layer $L+1$):** When $[A]$ appears again, $H_2$'s Query attends to $H_1$'s Key at $[B]$ and copies $[B]$ to the residual stream via its Value-Output pathway ($W_O W_V$).
- The abrupt macroscopic emergence of few-shot ICL during training directly coincides with the formation of these induction circuits.

## 2. Automated Circuit Discovery (ACDC)
Introduced by Conmy et al. (NeurIPS 2023), ACDC replaces manual activation patching with an algorithmic, greedy graph-pruning procedure:
1. Constructs the full computational directed acyclic graph (DAG) of all attention heads and MLP layers.
2. Performs systematic activation patching with corrupted baseline inputs.
3. Prunes non-essential edges iteratively whenever task loss deviation remains below tolerance $\tau$, isolating minimal faithful subcircuits (e.g., IOI circuit, Greater-Than circuit, docstring completion).

## 3. Scaling to Frontier Models: EAP & CD-T (2024–2025)
- **Edge Attribution Patching (EAP):** Replaces expensive patching forward passes with first-order gradient approximations ($g_e = \frac{\partial \mathcal{L}}{\partial a} \cdot \Delta a$), accelerating circuit discovery by $>100\times$.
- **Contextual Decomposition for Transformers (CD-T):** Algebraically disentangles non-linear feature interactions without gradient or forward-pass interventions, enabling instant circuit identification on 70B+ scale LLMs.
