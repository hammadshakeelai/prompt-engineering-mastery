# SimPO: Length-Normalized Margin Preference Alignment

- **Length-Normalized Reward**: Sequence reward is defined as average per-token log-likelihood $r(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y|x)$, eliminating verbosity bias.
- **Target Margin**: Enforces positive target margin $\gamma$ in Bradley-Terry loss, preventing marginal separations without reference model overhead.