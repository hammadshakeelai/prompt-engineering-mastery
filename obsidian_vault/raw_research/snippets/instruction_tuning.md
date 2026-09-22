# Instruction Following vs. Instruction Tuning

## Key Paradigms

### FLAN (Wei et al., 2021 - Google)
Multi-task instruction tuning across 62 NLP datasets grouped into 12 task clusters on LaMDA-PT (137B).
Proved instruction tuning significantly improves zero-shot performance on held-out task clusters.
Cross-task generalization is an emergent capability at scale.

### T0 and PromptSource (Sanh et al., 2021 - BigScience)
Trained T5 (11B) on extensive prompt diversity across varied task formatting templates.
T0 matches or surpasses zero-shot performance of GPT-3 (175B) on unseen tasks.
Key: prompt diversity > instance count per task.

### Super-NaturalInstructions (Wang et al., 2022 - Allen AI)
1,616 diverse NLP tasks across 76 task types in 55 languages with structured declarative instructions.
Scaling task and instruction diversity yields better OOD generalization than scaling instance count.

### InstructGPT (Ouyang et al., 2022 - OpenAI)
SFT -> Reward Model -> PPO on open-ended human intent alignment.
Human annotators prefer 1.3B aligned model over 175B unaligned base model.

## Key Principles
- Instruction Diversity: Forces meta-task of instruction execution, prevents mode collapse.
- Task Formatting Templates: Verbalize structured datasets into multiple phrasing variants, making representations invariant to prompt syntax.