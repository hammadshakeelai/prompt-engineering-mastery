# Online Direct Preference Optimization (Online DPO)

**Online DPO** bridges offline preference alignment and on-policy Actor-Critic reinforcement learning (PPO) by generating candidate completions on-the-fly and labeling them with an external reward model or verification oracle.

```mermaid
flowchart LR
    PROMPT["Prompt x"] --> POLICY["Active Policy π_θ"]
    POLICY --> GEN["Sample On-Policy Pair: y_1, y_2 ~ π_θ(·|x)"]
    GEN --> REWARD["Reward Oracle r_ϕ Scores & Labels Winner y_w / Loser y_l"]
    REWARD --> LOSS["Online DPO Step: -log σ(β log(π_w/π_ref) - β log(π_l/π_ref))"]
    LOSS --> UPDATE["Critic-Free Policy Update (Saves 50% VRAM over PPO)"]
```

## Mathematical Equivalence to PPO
Taking the gradient of the Online DPO objective:
$$\nabla_\theta \mathcal{L} = -\mathbb{E}\left[\sigma\left(\hat{r}_l - \hat{r}_w\right) \left(\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\right)\right]$$
The weight $\sigma(\hat{r}_l - \hat{r}_w)$ functions identically to a clipped advantage estimator $A(x, y)$ in PPO. By sampling on-policy, Online DPO eliminates offline distribution shift and reference policy collapse without requiring a value network (critic), cutting memory consumption by $50\%$ while matching or outperforming PPO and GRPO across reasoning benchmarks.

## Related Mechanics
- [[direct_preference_optimization_dpo]]
- [[reinforcement_learning_human_feedback]]
- [[orpo_odds_ratio_preference_optimization]]
- [[test_time_compute_scaling_and_rlvr_monograph]]
