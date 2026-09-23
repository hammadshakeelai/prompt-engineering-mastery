# DeepSeek-Prover-V1.5 & RMaxTS: Formal Theorem Proving (Xin et al., ICLR 2025)

## 1. The Sparse-Reward Bottleneck in Interactive Theorem Proving
Automated formal theorem proving in interactive environments like **Lean 4** requires generating sequences of tactic commands that pass strict kernel type-checking ($\vdash p : T$). Because an incomplete or slightly flawed proof receives a binary reward of zero, standard Monte Carlo Tree Search (MCTS) suffers from severe sample inefficiency and reward starvation.

## 2. Core Architectural Components of DeepSeek-Prover-V1.5

### 2.1 RMaxTS (Reward-Maximizing Tree Search)
To overcome sparse rewards, DeepSeek-Prover-V1.5 (arXiv:2408.08152) introduces **RMaxTS**, integrating the R-Max exploration heuristic into tree search:
- **Intrinsic State Novelty Reward:** When visiting an unfamiliar Lean 4 tactic state $s$, the model receives an intrinsic bonus:
  $$r_{\text{int}}(s) = \begin{cases} R_{\text{max}}, & \text{if } N(s) < M \\ 0, & \text{if } N(s) \ge M \end{cases}$$
  prioritizing unvisited proof states over known dead-ends.
- **Discounted Upper Confidence Bound (DUCB):** Tracks non-stationary value updates across tree search iterations $k \in [1, K]$ with discount factor $\gamma = 0.99$:
  $$N_\gamma(s, a) = \sum_{i=1}^k \gamma^{k - i} \mathbb{I}(s_i = s, a_i = a), \quad \bar{Q}_\gamma(s, a) = \frac{1}{N_\gamma(s, a)} \sum_{i=1}^k \gamma^{k - i} R_i(s, a)$$
  Selection uses PUCT over discounted statistics:
  $$a_t^* = \arg\max_{a} \left[ \bar{Q}_\gamma(s, a) + c \cdot P(s, a) \frac{\sqrt{\sum_{a'} N_\gamma(s, a')}}{1 + N_\gamma(s, a)} \right]$$

### 2.2 Truncate-and-Resume Feedback
When the Lean 4 REPL detects a syntax or typing error at tactic step $k$, generation is truncated immediately prior to the failure. The verified prefix $a_{1:k-1}$ and valid state $s_{k-1}$ are retained as prompt context for subsequent node expansion, avoiding costly full-trajectory restarts.

## 3. Empirical Results
- **miniF2F-test:** Achieves **63.5%** accuracy, outperforming whole-proof sampling baselines.
- **ProofNet:** Achieves **25.3%** on undergraduate-level mathematics theorems.
