# GRPO: Group Relative Policy Optimization (DeepSeek-R1)

- **Critic-Free Advantage**: Samples group of outputs $\{o_1, \dots, o_G\}$ per prompt. Computes advantage by normalizing rewards within the group:
  $$A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$$
- **Efficiency**: Eliminates the separate critic/value network, halving RL memory and compute overhead while eliciting self-reflection and reasoning in DeepSeek-R1.