# Math-Shepherd: Automated Process Supervision (Wang et al., ACL 2024)

## 1. Outcome vs. Process Supervision Pathologies
Outcome Reward Models (ORMs) assign a single terminal scalar $R \in \{0, 1\}$. This creates two failure modes in multi-step reasoning:
- **False Negatives:** Flawless intermediate reasoning that commits an arithmetic slip on the final step receives $R = 0$.
- **False Positives (Reward Hacking):** Flawed derivation paths that stumble on the correct numerical answer coincidentally receive $R = 1$.

## 2. Monte Carlo Step Quality Attribution
Math-Shepherd (Peiyi Wang et al., ACL 2024 / arXiv:2312.08935) eliminates manual human step annotation by assigning rewards via Monte Carlo rollouts:

1. **Empirical Rollout Ratio:**
   From intermediate reasoning step $s_t$, generate $M$ parallel continuation rollouts:
   $$r(s_t) = \frac{1}{M} \sum_{m=1}^M \mathbb{I}\left( \text{Terminal}(\tau_t^{(m)}) = y^* \right)$$
   Steps with higher probabilities of reaching the ground-truth answer $y^*$ receive higher supervision targets.

2. **PRM Training:**
   A classification head is trained on intermediate step delimiter tokens via cross-entropy loss against rollout targets $y_t \in [0, 1]$.

## 3. Test-Time Compute (TTC) & Policy Optimization
- **Best-of-$N$ Re-ranking:** Re-ranks candidate completions by cumulative step confidence $S(\tau) = \prod_{t=1}^T r_t$ or minimum bottleneck $S(\tau) = \min_t r_t$, beating ORMs by **$+5.2\%$ on GSM8K and $+3.8\%$ on MATH**.
- **Step-Level PPO:** Provides dense, non-sparse credit assignment for reinforcement learning, accelerating policy convergence.
