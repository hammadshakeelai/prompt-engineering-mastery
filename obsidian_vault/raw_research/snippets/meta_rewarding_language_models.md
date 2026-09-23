# Meta-Rewarding Language Models: Self-Improving Alignment (Wu et al., Meta / EMNLP 2024)

## 1. Saturation in Self-Rewarding LLMs
Self-Rewarding Language Models (Yuan et al., ICML 2024) train models to act as their own reward models using LLM-as-a-Judge prompting. However, self-rewarding saturates across 2–3 iterations because judgment capability fails to scale without external supervision.

## 2. LLM-as-a-Meta-Judge Architecture
Wu et al. (*Meta-Rewarding*, Meta / EMNLP 2024 / arXiv:2407.19594) introduce a second-order meta-evaluation loop:
1. **Judgment Generation:** Model acts as Judge, generating evaluations $J_1, J_2$ with rationales and scores on candidates $y_1, y_2$.
2. **Meta-Judging:** Model acts as Meta-Judge, evaluating which judgment is more factual, calibrated, and unbiased:
   $$M = \operatorname{Meta-Judge}(J_1, J_2 \mid x, y_1, y_2)$$
3. **Dual DPO Training:** Simultaneously fine-tunes judgment generation on preferred meta-judgments and policy instruction following on response pairs vetted by the updated judge.

## 3. Length Bias Filtering & Benchmark Gains
Applies quality-tier length thresholds ($\rho$) to prevent verbosity exploitation.
- **AlpacaEval 2:** Win rate improves from **$22.9\%$ to $39.4\%$** on Llama-3-8B-Instruct.
- **Arena-Hard:** Win rate increases from **$20.6\%$ to $29.1\%$** with zero human supervision.
