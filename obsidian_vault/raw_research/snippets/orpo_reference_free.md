# ORPO: Reference-Free Odds Ratio Preference Optimization

- **Unified Objective**: Combines SFT loss with an odds ratio penalty comparing generation odds of winning vs. losing responses in a single training step.
- **No Reference Policy**: Eliminates the frozen reference model required by DPO, significantly reducing VRAM footprint and compute costs.