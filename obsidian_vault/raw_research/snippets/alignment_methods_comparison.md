# Alignment Methods: RLHF vs RLAIF vs DPO vs GRPO

## RLHF (Ouyang et al. 2022 - InstructGPT)
- Pipeline: SFT -> Reward Model -> PPO with KL penalty against frozen reference model.
- Compute: Highest VRAM; maintains 4 models concurrently (Actor, Critic, Reference, Reward).
- Prone to reward hacking and training instability.
- Use for: High-stakes alignment requiring nuanced human judgment and safety boundaries.

## RLAIF (Bai et al. 2022 - Constitutional AI; Lee et al. 2023)
- Replaces human evaluators with frontier LLM annotators for preference scoring.
- Downstream policy optimized via PPO or DPO.
- Use for: Scalable alignment with limited human annotation budgets.

## DPO (Rafailov et al. 2023 - Direct Preference Optimization)
- Bypasses explicit RM and RL rollouts entirely.
- Optimizes Bradley-Terry preference objective via binary cross-entropy on chosen/rejected pairs.
- Compute: Lowest; loads only Policy and Reference models.
- Use for: General conversational alignment on static preference datasets.

## GRPO (Shao et al. 2024 - DeepSeekMath)
- Online RL discarding separate Critic/Value network.
- Samples group of outputs per prompt; normalizes rewards across group for relative advantages.
- ~50% VRAM savings vs PPO; requires high rollout inference compute.
- Use for: Verifiable tasks (math, coding) and RLVR self-reflection scaling.