# Speculative Decoding: Acceptance Rates & Draft Sizing

- **Acceptance Dynamics**: Probability governed by modified rejection sampling $\min(1, p_T/p_D)$. Higher in low-entropy regimes (code, grammar) and lower under high temperature.
- **Draft Model Sizing**: Optimal draft model is 10-50x smaller (~5-15% of target parameter count). Net speedup requires draft latency $t_D \ll t_T$ while sustaining acceptance rates $\ge 0.6-0.7$.