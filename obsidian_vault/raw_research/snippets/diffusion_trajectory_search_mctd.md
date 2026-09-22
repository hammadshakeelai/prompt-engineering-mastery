# Diffusion Trajectory Search & Monte Carlo Tree Diffusion (MCTD)

## 1. Limitations of Autoregressive MCTS
In standard autoregressive reasoning search (e.g., Tree-of-Thoughts, RAP, AlphaZero-LLM), search trees expand strictly along the temporal sequence dimension:
$$x_1 \to x_2 \to \dots \to x_T$$
- **Prefix Irreversibility:** Early reasoning decisions permanently constrain downstream expansions; mistakes early in a prefix cannot be revised without discarding entire subtrees.
- **Rollout Bandwidth Bottleneck:** Evaluating terminal states requires complete autoregressive token rollouts, creating heavy KV cache duplication across competing tree branches.

## 2. Denoising Tree Topology: Monte Carlo Tree Diffusion (MCTD)
Monte Carlo Tree Diffusion (MCTD, Yoon et al., NeurIPS 2024; C-MCTD, 2025) reconceptualizes planning by mapping MCTS to the reverse diffusion trajectory:
- **State Representation:** A node at tree depth $k$ represents an intermediate noisy/masked sequence $x_t$ across the *entire* sequence length $L$, where timestep $t \in [T, 0]$ decreases with tree depth.
- **Action Space:** Actions correspond to reverse stochastic transitions:
  $$x_{t-1} \sim q_\theta(x_{t-1} \mid x_t, c)$$
- **Holistic Value Estimation:** Unlike autoregressive value functions that must estimate returns from incomplete prefixes, diffusion value networks $V_\phi(x_t)$ evaluate the global structural coherence and constraint satisfaction across all token positions simultaneously.
- **Upper Confidence Bounds for Trees (UCT):** Node selection balances exploration and exploitation over continuous reverse diffusion branches:
  $$\text{Score}(x_{t-1}) = Q(x_{t-1}) + c_{\text{puct}} P_\theta(x_{t-1} \mid x_t) \frac{\sqrt{N(x_t)}}{1 + N(x_{t-1})}$$

## 3. Comparative Advantages for Complex Reasoning
- **Non-Causal Global Revision:** If a downstream constraint conflict emerges during denoising, MCTD can prune the trajectory branch and explore alternative denoising paths that modify early and late tokens simultaneously.
- **Test-Time Compute (TTC) Scaling:** Test-time budget $B$ expands search depth and branch width along the denoising dimension, decoupling search quality from generation length.
- **Compositional Stitching (C-MCTD):** Sub-trajectories generated across independent diffusion segments can be composed and verified, enabling planning horizons that extend far beyond model pretraining contexts.
