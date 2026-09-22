# Cutting-Edge Prompt Optimization (2025 Frontiers)

## AutoPDL (Spiess et al., 2025 - arXiv:2504.04365)
Frames LLM agent prompting as AutoML problem over combinatorial space of agentic patterns (Zero-Shot, CoT, ReAct, ReWOO) and demonstrations. Uses successive halving to efficiently navigate the space. Solutions are human-readable, editable PDL programs. Evaluations across 3 tasks and 7 LLMs show +9.21 avg percentage point gains (up to +67.5pp).

## PromptWizard (Agarwal et al., 2024 - arXiv:2405.18369 - Microsoft)
Fully automated discrete prompt optimization via self-evolving, self-adapting mechanism. Feedback-driven critique and synthesis balances exploration and exploitation. Iteratively refines both instructions AND in-context examples. Superior performance across 45 tasks. Significant reduction in API calls vs. competing methods.

## GAAPO (2025 - arXiv:2504.07157)
Genetic Algorithm Applied to Prompt Optimization. Hybrid framework evolving prompts through successive generations. Multiple generation methods as distinct genetic operators: direct mutation, crossover, lamarckian adaptation. Wider prompt space exploration than single-algorithm methods like PromptBreeder.

## LatentPrompt (Bystronski et al., 2025 - arXiv:2508.02452)
Prompt optimization in continuous latent semantic space. Embeds seed prompts, systematically explores latent space to identify accuracy-maximizing prompts without hand-crafted rules. Model-agnostic (black-box LLM access + evaluation metric only). +3% accuracy on Financial PhraseBank after one optimization cycle.

## promptolution (Zehle et al., EACL 2026)
Modular, lightweight open-source framework for automatic prompt optimization. Includes CAPO (Cost-Aware Prompt Optimization): GA-based approach using AutoML techniques, jointly optimizing instructions and cost-efficiency. Supports multiple LLM interfaces and evaluation metrics.