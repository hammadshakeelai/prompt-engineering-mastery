# Multi-Agent Debate & Bradley-Terry Elo Jury Consensus (Du et al., ICML 2024)

## 1. Premise Entrapment in Single-Agent Autoregression
Single-agent generation suffers from autoregressive confirmation bias: early reasoning errors become self-reinforcing context, trapping the model in fallacious derivation paths.

## 2. Multi-Agent Debate Dynamics
Multi-Agent Debate (Du et al., MIT / ICML 2024 / arXiv:2305.14325) initializes $N$ heterogeneous agents $\{\pi_i\}_{i=1}^N$ with diverse personas. Across synchronous rounds $t \in \{1, \dots, T\}$, each agent updates its response conditioned on peer transcripts:
$$y_i^{(t)} \sim \pi_i(y \mid x, y_i^{(t-1)}, \mathcal{M}_{-i}^{(t-1)})$$
- **Verification-Generation Asymmetry:** Identifying a derivation flaw requires lower epistemic entropy than inventing a novel proof. Truth functions as a game-theoretic basin of attraction, driving hallucination variance $\operatorname{Var}(\bar{\epsilon}) \to 0$.

## 3. Confidence-Weighted Elo Jury Aggregation
An impartial referee model updates agent skill ratings $R_i$ via Bradley-Terry preference modeling across debate transcripts. Consensus is computed via confidence-weighted Elo voting:
$$w_i = c_i \cdot \frac{\exp(R_i / \tau)}{\sum_k \exp(R_k / \tau)}, \quad \hat{y} = \operatorname{argmax}_y \sum_{i=1}^N w_i \cdot \mathbb{I}(\phi(y_i^{(T)}) = y)$$
Lifts accuracy by **$+8.4\%\text{--}+14.2\%$** on GSM8K and MMLU while cutting hallucinations by $>60\%$.
