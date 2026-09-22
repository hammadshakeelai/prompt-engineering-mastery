# Active Prompting (Diao et al., ACL 2024)

**Citation:** arXiv:2302.12246

## Core Concept
Applies active learning to select which questions to annotate with CoT rationales, targeting highest-uncertainty examples.

## Four-Stage Pipeline
1. **Uncertainty Estimation:** Query LLM k times per question with temperature > 0.
2. **Metric Quantification:** Measure disagreement (distinct answers ratio) or entropy of prediction distributions.
3. **Selective Annotation:** Human annotators write CoT rationales for top-N most uncertain questions.
4. **Inference:** Annotated high-uncertainty exemplars used as few-shot prompt.

## Results
- Outperforms standard CoT, random selection, and self-consistency on GSM8K, SVAMP, ASDiv, StrategyQA.
- Uncertainty strongly correlates with model error, ensuring exemplars cover boundary conditions.