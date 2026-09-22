import os

base = r'c:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\raw_research\snippets'
os.makedirs(base, exist_ok=True)

files = {
'coding_agents.md': """# Agentic AI Coding Systems: SWE-agent, Devin, and OpenHands (2024)

## Architecture & Problem Solving
Autonomous coding agents resolve repository-level GitHub issues through iterative ReAct/CodeAct loops:
1. **Context & Localization**: Ingesting issue descriptions and searching codebases to isolate fault locations.
2. **Reproduction & Verification**: Generating reproducing scripts, applying edits, and executing test suites to verify regressions before finalizing git patches.

Systems diverge in architecture: **SWE-agent** pioneered dedicated Agent-Computer Interfaces (ACI) for LM-centric navigation; **Devin** introduced a managed environment with shell, browser, and persistent planner; **OpenHands** established open-source Docker sandboxing using executable Python/bash (CodeAct).

## Tool Use Patterns
- **Search & Navigation**: High-level commands (`find_file`, `search_dir`, ripgrep) return condensed, token-budgeted directory trees to prevent context overflow.
- **Code Editor / ACI**: Windowed viewers (~100 lines with scrolling/offset) paired with surgical line-replacement or pattern-matching editors, pre-validated via linting.
- **Bash & Sandboxed Execution**: Persistent Linux shells run commands and invoke pytest. **Claude 3.5 Sonnet Computer Use** extended this to OS GUIs, terminals, and browsers via screen perception.

## Performance on SWE-Bench (2024)
- Early 2024: Raw LLMs scored <2-4%. Devin reported **13.86%** unassisted resolve rate; SWE-agent achieved **~12.5%** (Full) and **~18%** (Lite) with GPT-4.
- Late 2024 (SWE-bench Verified): Claude 3.5 Sonnet hit **49.0%**, while **OpenHands (CodeAct v2.1)** achieved **53.0%**.""",

'red_teaming.md': """# Constitutional AI Red-Teaming & Automated Adversarial Testing

## Perez et al. (2022) & Automated Generation
Perez et al. (2022) pioneered using language models to red-team other language models (*"Red Teaming Language Models with Language Models"*). A generator LM crafts adversarial prompts to surface toxic outputs, private data leakage, and harmful edge cases.
- **Zero-shot / Few-shot Generation:** Prompting the red-team LM with instructions or seeded examples of harmful queries.
- **Reinforcement Learning (RL):** Training the adversary model with RL to maximize harmfulness scores produced by a safety classifier while applying a KL penalty to preserve fluency.

## Anthropic's Red Team Findings
- **Scaling Behaviors (Ganguli et al. 2022):** Larger, RLHF-aligned models become substantially harder to jailbreak via naive prompts, demanding more complex, multi-turn adversarial attacks.
- **Vulnerability Patterns:** Models exhibit sycophancy, persona adoption biases, and subtle safety degradation across multi-turn interactions.
- **Constitutional AI Integration (Bai et al. 2022):** Models generate adversarial prompts, produce responses, critique them against constitutional principles, and revise them (RLAIF).

## Automated vs. Human Red-Teaming
- **Throughput & Coverage:** Automated testing generates hundreds of thousands of diverse attacks rapidly, uncovering long-tail failure modes and boundary vulnerabilities.
- **Depth vs. Breadth:** Human red-teamers excel at nuanced social engineering and zero-day conceptual jailbreaks; automated testing excels at systematic regression testing.""",

'agent_planning.md': """# Agent Planning and Task Decomposition

## Taxonomy & Core Mechanisms (Huang et al., 2024 Survey)
Huang et al. (2024) categorize LLM agent planning into task decomposition, plan selection, external modules, reflection, and memory:
- **Subgoal Generation:** Decomposes high-level goals into DAGs or linear action sequences, dynamically adapting steps based on environment feedback.
- **Hierarchical Task Networks (HTN):** Recursively breaks compound tasks into intermediate subtasks and primitive actions, enabling multi-level abstraction.
- **PDDL-Style Neuro-Symbolic Planning:** LLM translates unstructured descriptions into formal Planning Domain Definition Language (PDDL), allowing sound symbolic solvers to guarantee plan validity.

## Key Foundational Frameworks
- **Inner Monologue (Huang et al., 2022):** Pioneer of closed-loop embodied planning. Integrates multimodal environmental feedback (success detection, scene descriptions) into continuous language prompts for dynamic re-planning without fine-tuning.
- **Voyager (Wang et al., 2023):** Lifelong embodied learning agent in Minecraft:
  1. *Automatic Curriculum:* Progressively generates open-ended subgoals tailored to current state.
  2. *Skill Library:* Synthesizes and self-refines reusable executable JavaScript routines, indexed via vector embeddings.
  3. *Iterative Prompting:* Incorporates compiler errors and environmental feedback for self-correction.""",

'rag_evaluation.md': """# RAGAS: Evaluation Framework for RAG (Es et al., 2023)

The **RAGAS** (Retrieval Augmented Generation Assessment) framework evaluates RAG pipelines reference-free by decoupling retrieval from generation.

## Core Metrics
### 1. Retrieval Quality
- **Context Precision:** Evaluates signal-to-noise ratio and ranking. Measures whether relevant chunks are ranked higher using mean average precision.
- **Context Recall:** Quantifies the proportion of reference/ground-truth statements successfully retrieved in the context.

### 2. Generation Quality
- **Faithfulness:** Quantifies factual grounding to eliminate hallucinations. Computes the ratio of atomic claims in generated response directly deducible from context.
- **Answer Relevance:** Synthesizes questions from the generated response and measures mean cosine similarity against original query.

## Comparison to Generic LLM-as-a-Judge
Standard LLM-as-a-judge approaches prompt an LLM for holistic 1-5 scores, introducing verbosity and position biases. RAGAS improves via:
1. **Atomic Decomposition:** Decomposes responses into atomic claims rather than subjective grading.
2. **Deterministic Aggregation:** Programmatic mathematical formulas over verified claims.
3. **Reference-Free Monitoring:** Automated evaluation of live systems without human-annotated references.""",

'knowledge_distillation.md': """# Knowledge Distillation from LLMs for Smaller Models

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
- **LLaMA Distillation**: Distilling Llama 3.1 405B into 8B/70B using synthetic CoT and torchtune logit recipes.""",

'cot_when_it_helps.md': """# Think Step by Step vs. Direct Answering: When CoT Helps and Hurts

## When CoT Helps
- **Complex Multi-Step & Math Reasoning:** Multi-hop logical deductions, symbolic manipulation, and algorithmic math (GSM8K, MultiArith).
- **Working Memory & Decomposition:** Emitting intermediate tokens acts as an external computational scratchpad, breaking compound state transitions into tractable single steps.

## When CoT Hurts
- **Simple & Intuitive Tasks:** Straightforward factual recall, sentiment analysis, or pattern-matching suffer from "reasoning inflation" and spurious failure surfaces.
- **Error Cascades:** An early hallucination or arithmetic error compounds down the chain, derailing an otherwise obvious answer.
- **Latency & Cost Overhead:** High prompt instability and token inflation with zero or negative accuracy return on non-reasoning tasks.

## Key Empirical Studies
- **Kojima et al. (2022):** Appending *"Let's think step by step"* surged MultiArith accuracy from 17.7% to 78.7% and GSM8K from 10.4% to 40.7%.
- **Fu et al. (2022):** Reasoning performance scales with chain complexity; selecting exemplars with more reasoning steps outperformed standard few-shot CoT.
- **Sprague et al. (2024):** Survey showing CoT benefits are predominantly confined to mathematical and symbolic domains; on commonsense tasks, direct answering matches or exceeds CoT."""
}

written = 0
for fname, content in files.items():
    path = os.path.join(base, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    written += 1

print(f'Successfully wrote {written} snippet files.')
