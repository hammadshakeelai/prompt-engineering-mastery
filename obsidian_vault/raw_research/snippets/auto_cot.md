# Automatic Chain-of-Thought (Auto-CoT) (Zhang et al., ICLR 2023)

Citation: arXiv:2210.03493

## Core Concept
Automates few-shot CoT demonstration generation via semantic question clustering + zero-shot rationale generation.
Eliminates manual exemplar authoring.

## Two-Stage Pipeline
1. Question Clustering: Embeds dataset questions with Sentence-BERT, partitions into k clusters (k=8) via k-means to maximize semantic and structural diversity.
2. Demonstration Sampling and Generation:
   - Selects representative candidate from each cluster (proximity to centroid, heuristic filters on length and step count).
   - Generates step-by-step rationales via Zero-Shot-CoT ("Let's think step by step").
   - Combines k auto-constructed (Question, Rationale, Answer) tuples into few-shot prompt.

## Comparison to Manual CoT
- Zero Human Effort: Manual CoT requires labor-intensive task-specific engineering.
- Diversity: Clustering prevents overfitting to specific error patterns, resilient to occasional zero-shot mistakes.
- Performance: Matches or exceeds Manual-CoT on 10 benchmarks (GSM8K, SVAMP, MultiArith, StrategyQA).