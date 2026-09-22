# Knowledge Distillation from LLMs for Smaller Models

Knowledge distillation (KD) compresses capabilities from large teacher LLMs into compact student models.

## 1. Output Logit Distillation
When teacher logits are accessible, student minimizes KL divergence against teacher's softened probabilities (temperature $T$). MiniLLM applies reverse-KL divergence with token-level policy gradients to alleviate exposure bias and student mode collapse.

## 2. White-Box Distillation: Intermediate Representations
- **Hidden State Alignment**: Minimizes MSE or cosine distance between teacher and student layer outputs using linear projections.
- **Attention Map Distillation**: Transfers multi-head self-attention distributions across corresponding layers, preserving contextual token-relational geometry.

## 3. Black-Box Distillation: Chain-of-Thought Traces
When weights/logits are inaccessible, prompting teachers to output step-by-step CoT rationales transfers structured reasoning. Students fine-tune on rationale-answer pairs (Distilling Step-by-Step, Orca).

## 4. Key Examples (2023–2024)
- **Lion (Jiang et al., 2023)**: Adversarial framework (Imitation -> Discrimination -> Generation) allowing a 7B student to rival 13B models.
- **DistillSpec**: Aligns speculative decoding draft models via on-policy logit distillation, yielding 10-45% inference acceleration.
- **LLaMA Distillation**: Distilling Llama 3.1 405B into 8B/70B using synthetic CoT and torchtune logit recipes.