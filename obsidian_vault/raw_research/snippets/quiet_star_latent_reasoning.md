# Quiet-STaR: Continuous Latent Reasoning (Zelikman et al., ICML 2024)

## 1. Generalizing Reasoning to Continuous Pretraining
Traditional chain-of-thought (CoT) reasoning occurs only when triggered by explicit instructions or specialized prompt structures. Quiet-STaR (Eric Zelikman et al., Stanford / ICML 2024 / arXiv:2403.09629) demonstrates that language models can learn to generate **latent rationales at every token position** during standard pretraining without task-specific data.

## 2. Architecture & Algorithmic Mechanics
1. **Parallel Rationale Generation:**
   Inserts $T$ internal thought tokens $t_1, \dots, t_T$ between sequence tokens $x_i$ using tokenwise parallel sampling.
2. **Learned Mixing Head:**
   Interpolates between thought-augmented predictions and baseline direct predictions:
   $$P_{\text{mix}}(x_{i+1}) = \alpha_i \cdot P_\theta(x_{i+1} \mid x_{\le i}, t_{1:T}) + (1 - \alpha_i) \cdot P_\theta(x_{i+1} \mid x_{\le i})$$
   where $\alpha_i = \sigma(W_{\text{mix}} h_i)$.
3. **Non-Myopic Horizon REINFORCE Reward:**
   Thoughts are rewarded based on their ability to improve prediction across future token horizons $x_{i+1:i+n}$:
   $$R_i = \sum_{j=1}^n \left( \log P_\theta(x_{i+j} \mid x_{\le i}, t_{1:T}) - \log P_\theta(x_{i+j} \mid x_{\le i}) \right)$$
   Policy gradients update rationale generation parameters using a leave-one-out baseline.

## 3. Empirical Results
- Improves zero-shot GSM8K from **$5.9\%$ to $10.9\%$** on open base models.
- Boosts CommonsenseQA from **$36.3\%$ to $47.2\%$** without task-specific fine-tuning.
