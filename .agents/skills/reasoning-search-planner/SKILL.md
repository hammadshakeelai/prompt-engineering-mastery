---
name: reasoning-search-planner
description: Specialized directive for Test-Time Compute (TTC) scaling, Monte Carlo Tree Search (MCTS), Monte Carlo Tree Diffusion (MCTD), Process Reward Model (PRM) guided search, step-level beam search, Best-of-N, and verification loops.
---

# Reasoning Search Planner Skill

Use this skill when designing test-time compute (TTC) reasoning architectures, implementing tree-search reasoning loops (MCTS, MCTD, Beam Search), integrating process reward models (PRMs), or optimizing inference-time exploration vs exploitation for frontier reasoning systems.

## 1. Test-Time Compute (TTC) Scaling Laws

1. **Sequential vs. Parallel Compute Allocation:**
   - **Parallel Scaling (Best-of-$N$, Self-Consistency):** Scales compute by drawing $N$ independent trajectories and aggregating via majority voting, verifier scoring, or semantic clustering. Saturates when problem difficulty exceeds the base policy's single-pass support.
   - **Sequential Scaling (Long CoT, Iterative Refinement):** Allocates compute to deeper chain-of-thought tokens, intermediate self-correction, and error backtracking.
   - **FLOP-Optimal Allocation:** On complex algorithmic or theorem-proving tasks, allocate compute hybridly: spend $60\%\text{--}70\%$ of test-time FLOPs on tree-based branching/verification and $30\%\text{--}40\%$ on sequential depth.

2. **Process Supervision & Verifier Integration:**
   - Use **Process Reward Models (PRMs)** (e.g., Math-Shepherd) rather than pure Outcome Reward Models (ORMs). Evaluate each intermediate reasoning step $s_t$:
     $$r_t = \text{PRM}(s_{1:t})$$
   - Prune trajectories immediately when step confidence $r_t < \tau_{\text{threshold}}$, preventing catastrophic error propagation early in the search tree.

## 2. Tree-Search Reasoning Topologies

1. **Autoregressive Monte Carlo Tree Search (MCTS):**
   - **Selection:** Traverse the reasoning tree using Upper Confidence Bounds for Trees (UCT / PUCT):
     $$a^* = \operatorname{argmax}_a \left[ Q(s, a) + c_{\text{puct}} P_\theta(a \mid s) \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]$$
   - **Expansion & Rollout:** Generate $K$ step-level candidates using temperature-scaled sampling. Terminate rollouts via PRM value evaluation rather than expensive full-trajectory generation.
   - **Backpropagation:** Update visitation counts $N(s, a)$ and action-value estimates $Q(s, a)$ along the trajectory path.

2. **Monte Carlo Tree Diffusion (MCTD) for Trajectory Planning:**
   - On tasks requiring global structural coordination (code refactoring, multi-constraint planning), employ reverse-diffusion tree search.
   - Expand tree branches along the reverse denoising timestep $x_t \to x_{t-1}$.
   - Evaluate full candidate trajectories holistically at each depth, bypassing autoregressive prefix irreversibility.

## 3. Aggregation, Clustering & Calibration

1. **AlphaCode-Style Clustering:**
   - For code and programmatic reasoning, do not aggregate via naive string matching.
   - Cluster generated solutions by executing generated test cases or AST fingerprinting. Select the representative from the largest non-empty passing cluster.
2. **Semantic Entropy Uncertainty Filtering:**
   - Group reasoning answers into semantic equivalence clusters $\mathcal{C}_k$.
   - Calculate epistemic entropy: $\text{SE} = -\sum_k p(\mathcal{C}_k) \log p(\mathcal{C}_k)$. Reject or trigger deeper tree exploration when $\text{SE}$ exceeds calibrated confidence boundaries.
