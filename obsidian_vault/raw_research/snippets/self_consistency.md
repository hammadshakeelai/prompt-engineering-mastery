# Self-Consistency in CoT (Wang et al., 2022)

- **Mechanism**: Samples diverse reasoning paths via temperature sampling ($T=0.5-0.7$) and takes unweighted majority vote over answers.
- **Sample Size**: Steepest gains occur at $k=5-10$; marginal returns diminish toward $k=40$.
- **Calibration**: Consensus vote share acts as an unsupervised, well-calibrated confidence score.
- **Benchmarks**: +10.6% to +17.9% GSM8K, +11% to +14% SVAMP, up to +24% MultiArith (reaching 99.3%).