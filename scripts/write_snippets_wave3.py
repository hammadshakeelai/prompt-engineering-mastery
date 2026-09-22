import os

base = r'c:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\raw_research\snippets'
os.makedirs(base, exist_ok=True)

files = {
'toolformer_gorilla.md': """# Tool-Augmented Language Models: Toolformer and Gorilla

## 1. Toolformer (Schick et al., 2023)
Self-supervised learning allowing LMs to teach themselves how/when to use external tools without human annotations.
- Candidate Generation: Model samples candidate API calls via few-shot prompts.
- Execution and Filtering: Calls external APIs (calculator, QA, Wikipedia, calendar, translation). Retains calls only if conditioning on API output reduces cross-entropy loss on future tokens.
- Finetuning: Fine-tunes base LM (GPT-J 6B) on filtered trajectories, learning autonomous tool invocation.

## 2. Gorilla (Patil et al., 2023)
LLaMA-based model fine-tuned to write correct executable API calls from natural language prompts.
- APIBench: Evaluated across 1600+ real-world APIs (TorchHub, HuggingFace, TensorHub). Outperforms GPT-4 and Claude in API generation accuracy.
- Generates complex multi-argument API requests while respecting schema constraints and API updates.

## 3. Retrieval-Aware Training (RAT)
Embeds retriever directly into the fine-tuning curriculum.
- Model conditioned on retrieved (and noisy/distractor) API docs during training.
- Forces LM to ground syntax and arguments in retrieved docs rather than stale parametric memory.
- Zero-shot generalization to unseen/modified APIs by updating the retriever index.""",

'p_tuning.md': """# Soft Prompts and P-Tuning (Liu et al.)

## Discrete vs. Continuous Soft Prompts
Discrete prompts constrained to vocabulary tokens, highly sensitive to phrasing.
Soft prompts: virtual continuous embedding vectors optimized directly via gradient descent.
Unconstrained by dictionary, discover latent representations tailored to downstream objectives.

## P-Tuning v1 (Liu et al., 2021 - "GPT Understands, Too")
- Injects continuous virtual prompt tokens into input embedding layer.
- Uses BiLSTM + MLP prompt encoder to capture dependencies and stabilize optimization.
- Primarily boosts NLU and few-shot knowledge probing on autoregressive models.

## P-Tuning v2 (Liu et al., 2022 - Deep Prompt Tuning)
- Inserts learnable prefix vectors into Key and Value matrices at EVERY self-attention layer.
- Deep injection dramatically increases representational capacity.
- Matches full fine-tuning performance across model scales 0.3B-100B+ and complex tasks (NER).

## Computational Advantages
- Trains only 0.1-3% task-specific parameters; freezes >99% of backbone.
- Single frozen model serves hundreds of tasks by swapping lightweight prefix weights.
- Pre-computed prefix embeddings cacheable in KV-cache for rapid inference.""",

'needle_in_haystack.md': """# Needle in a Haystack (NIAH) Benchmark

Introduced by Greg Kamradt (late 2023). Tests LLM long-context retrieval by hiding a specific fact (needle) in background text (haystack, typically Paul Graham essays).

## Methodology
Two-axis 2D retrieval heatmap:
1. Context Length: 1k to 128k+ tokens.
2. Document Depth: Needle position from 0% (top) to 100% (bottom) of context.

## Key Model Results
- GPT-4 Turbo: Reliable to ~64k tokens, degradation toward 128k especially at mid-depths.
- Claude 2.1 (200k): Pronounced lost-in-the-middle effect at 50k-150k mid-depths. Claude 3 (Opus/Sonnet) resolved this, achieving >99% recall across 200k.
- Gemini 1.5 Pro: Near-flawless (>99%) recall across 1M-2M token contexts across all depths and modalities.

## Implications for Prompt Design
1. Place critical instructions at beginning OR end, not buried in the middle.
2. Prompt Scaffolding: Ask model to quote exact context to improve attention anchoring.
3. Selective Context: NIAH tests simple lookup, not multi-hop reasoning. Targeted RAG > blind context dumping.""",

'alignment_methods_comparison.md': """# Alignment Methods: RLHF vs RLAIF vs DPO vs GRPO

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
- Use for: Verifiable tasks (math, coding) and RLVR self-reflection scaling.""",

'reward_model_design.md': """# Reward Model Design for RLHF: Process vs. Outcome

## ORMs vs. PRMs
- Outcome Reward Models (ORMs): Evaluate only the terminal output. Simple annotation, but process-blind - reward correct final answers even via flawed intermediate reasoning.
- Process Reward Models (PRMs): Step-level feedback scoring each individual reasoning step. Granular credit assignment, better for MCTS/Best-of-N search. Require high annotation overhead.

## PRM800K Dataset (Lightman et al. 2023 - OpenAI "Let's Verify Step by Step")
- 800,000 human step-level ratings across 75,000 solutions on MATH benchmark.
- Proved process supervision substantially outperforms outcome supervision for complex reasoning.
- Foundation for step-level verification showing denser supervision produces more consistent reasoning.

## Reward Model Quality as RLHF Bottleneck
- Goodhart's Law: RL optimization exploits RM blind spots (length, format, sycophancy) rather than acquiring true capabilities.
- Distributional Shift: As policy drifts OOD, RMs output uncalibrated heavy-tailed positive scores.
- Objective Compression: Collapsing multidimensional human preferences into scalars discards vital nuance.""",

'cot_faithfulness.md': """# Chain-of-Thought Faithfulness Evaluation Metrics

## Faithfulness vs. Post-Hoc Rationalization
Faithful rationale: reflects the model's actual internal decision-making.
Post-hoc rationalization: model selects answer independently, then generates explanation to justify it.
Measuring faithfulness requires counterfactual perturbations, not surface plausibility checks.

## Lanham et al. (2023): Truncation and Intervention Tests
- Early Answering: Cuts CoT at intermediate steps; if accuracy unchanged, model didn't causally depend on subsequent reasoning.
- Finding: Larger, more capable models frequently exhibit DECREASED faithfulness, solving tasks via direct internal computation while generating post-hoc narrative traces.

## Turpin et al. (2023): Biased Reasoning
- Injected biasing cues (reordered options so A is always right, or prepended user opinion).
- Models were swayed (accuracy dropped up to 36% on BIG-Bench Hard).
- Models failed to mention biasing features in CoT - confabulated plausible neutral rationales instead.
- CoT can actively conceal true decision drivers.

## Parcalabescu and Frank (2023): CC-SHAP
- Standard faithfulness metrics only measure output-level self-consistency, not faithfulness to internal operations.
- Introduced CC-SHAP to measure internal feature attributions across both explanation and prediction.
- True faithfulness requires inspecting internal representations, not just output text alignment.""",

'auto_cot.md': """# Automatic Chain-of-Thought (Auto-CoT) (Zhang et al., ICLR 2023)

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
- Performance: Matches or exceeds Manual-CoT on 10 benchmarks (GSM8K, SVAMP, MultiArith, StrategyQA).""",

'agent_memory_taxonomy.md': """# LLM Agent Memory Taxonomy (CoALA Framework)

Draws from cognitive psychology to organize agent memory into four functional tiers.

## 1. Working Memory (Short-Term Workspace)
- Cognitive Basis: Baddeley and Hitch's Working Memory Model (1974) - active workspace for information under cognitive load.
- Implementation: LLM context window and runtime scratchpads. Holds conversational history, active CoT reasoning steps, sub-goals, tool call inputs/outputs.

## 2. Episodic Memory (Autobiographical Experience)
- Cognitive Basis: Endel Tulving (1972) - explicit memory of specific past events and agent-user interactions.
- Implementation: Vector Databases (Chroma, Qdrant) with dense embeddings + temporal metadata. Retrieved via k-NN semantic similarity, recency decay, importance scoring.

## 3. Semantic Memory (Factual and Conceptual Knowledge)
- Cognitive Basis: Tulving's declarative taxonomy - generalized world knowledge decoupled from specific episodes.
- Implementation: Knowledge Graphs, Graph RAG, relational entity stores. Deterministic entity-relation traversal and structured fact retrieval.

## 4. Procedural Memory (Implicit Skills and Policies)
- Cognitive Basis: Larry Squire's non-declarative memory (1987) - automated how-to knowledge.
- Implementation: Encoded into model weights via SFT/LoRA, RLHF/DPO. Operationalized via system prompt instructions, few-shot tool exemplars, deterministic execution workflows.""",

'cutting_edge_2025.md': """# Cutting-Edge Prompt Optimization (2025 Frontiers)

## AutoPDL (Spiess et al., 2025 - arXiv:2504.04365)
Frames LLM agent prompting as AutoML problem over combinatorial space of agentic patterns (Zero-Shot, CoT, ReAct, ReWOO) and demonstrations. Uses successive halving to efficiently navigate the space. Solutions are human-readable, editable PDL programs. Evaluations across 3 tasks and 7 LLMs show +9.21 avg percentage point gains (up to +67.5pp).

## PromptWizard (Agarwal et al., 2024 - arXiv:2405.18369 - Microsoft)
Fully automated discrete prompt optimization via self-evolving, self-adapting mechanism. Feedback-driven critique and synthesis balances exploration and exploitation. Iteratively refines both instructions AND in-context examples. Superior performance across 45 tasks. Significant reduction in API calls vs. competing methods.

## GAAPO (2025 - arXiv:2504.07157)
Genetic Algorithm Applied to Prompt Optimization. Hybrid framework evolving prompts through successive generations. Multiple generation methods as distinct genetic operators: direct mutation, crossover, lamarckian adaptation. Wider prompt space exploration than single-algorithm methods like PromptBreeder.

## LatentPrompt (Bystronski et al., 2025 - arXiv:2508.02452)
Prompt optimization in continuous latent semantic space. Embeds seed prompts, systematically explores latent space to identify accuracy-maximizing prompts without hand-crafted rules. Model-agnostic (black-box LLM access + evaluation metric only). +3% accuracy on Financial PhraseBank after one optimization cycle.

## promptolution (Zehle et al., EACL 2026)
Modular, lightweight open-source framework for automatic prompt optimization. Includes CAPO (Cost-Aware Prompt Optimization): GA-based approach using AutoML techniques, jointly optimizing instructions and cost-efficiency. Supports multiple LLM interfaces and evaluation metrics.""",
}

written = 0
for fname, content in files.items():
    path = os.path.join(base, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    written += 1

print(f'Successfully wrote {written} snippet files.')
