# Synthetic Data Generation for LLM Training

## 1. Self-Instruct (Wang et al., 2022)
Bootstraps instruction-following datasets via iterative semi-automated pipeline.
- Seed Tasks: 175 human-written task exemplars.
- Generation and Filtering: LLM (GPT-3) generates instruction-input-output triplets. Pruned via ROUGE-L similarity (<0.7) and heuristic filters.
- Impact: Produced 52K instructions, blueprint for cost-effective synthetic alignment.

## 2. Stanford Alpaca (Taori et al., 2023)
- Used text-davinci-003 to generate 52,000 instruction-response pairs from 175 seed tasks for under $500.
- Fine-tuned LLaMA-7B on synthetic dataset, proving lightweight open models can approximate proprietary instruction-following behaviors.

## 3. WizardLM: Evol-Instruct (Xu et al., 2023/2024)
Scales prompt complexity via iterative evolution:
- In-Depth Evolution: Adds constraints, deepens reasoning, concretizes abstract concepts, chains complex logic.
- In-Breadth Evolution (Mutation): Broadens topic coverage, creates novel domain tasks.
- Enables models to master sophisticated coding and mathematical reasoning.

## 4. Orca 2: Synthetic CoT and Strategy Selection (Mitra et al., 2023)
Teaches Small Language Models diverse reasoning strategies (step-by-step, recall-then-generate, direct answering).
- Prompt Erasure: Teacher (GPT-4) gets rich system prompts guiding detailed CoT traces. During student training, guidance prompt is stripped. Student must autonomously synthesize internal reasoning paths.