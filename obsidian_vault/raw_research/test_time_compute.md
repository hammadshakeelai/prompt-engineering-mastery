# Frontier Reasoning Models: Test-Time Compute, RLVR, and Prompting

## 1. Test-Time Compute (TTC) Scaling Laws
- **Core Mechanism**: A paradigm shift where scaling laws transition from "train-time" parameter/data saturation to "test-time" inference computation. Model performance improves monotonically by allocating additional compute to internal reasoning processes before outputting a final answer.
- **Scaling Modalities**:
  - **Sequential Scaling**: Increasing the linear length of the Chain-of-Thought (CoT). The model executes deeper sequential exploration to resolve bottlenecks.
  - **Parallel Scaling**: Generating multiple concurrent reasoning paths via search algorithms (e.g., Monte Carlo Tree Search, Beam Search, Majority Voting) and selecting the optimal result.
- **Diminishing Returns & Saturation Limits**: Extending compute indefinitely yields diminishing returns ("no free lunch"). Performance depends heavily on the ratio of compute budget to problem difficulty. Over-computing simple tasks degrades efficiency; complex logic requires dynamic, adaptive inference budgets.
- **Economics & Architecture**: DeepSeek-R1 and OpenAI o1 allocate internal computation by generating 10–100x more tokens *per query* during the reasoning phase, significantly pivoting global AI compute spend toward inference over pre-training.

## 2. Reinforcement Learning with Verifiable Rewards (RLVR)
- **Definition**: A deterministic training framework that replaces human-labeled subjective preferences (RLHF) with programmatic, binary verification (e.g., mathematical accuracy, code compilation, unit test success, logic constraints).
- **Algorithm (GRPO)**: DeepSeek-R1 relies heavily on **Group Relative Policy Optimization (GRPO)**. GRPO samples a group of completions for a single prompt and computes rewards based on relative performance within the cohort.
  - *Architectural Limit*: GRPO completely eliminates the need for an expensive, memory-heavy "critic" or "value" neural network, radically lowering training overhead.
- **Emergent Reasoning Properties**: By rewarding only the verified final answer, the model is forced to autonomously discover optimal problem-solving heuristics. This triggers emergent "System 2" behaviors, including mid-thought self-correction, backtracking, and hypothesis verification without supervised step-by-step data.
- **Domain Limitations**: RLVR is strictly limited to objective domains. For subjective tasks (creative writing, nuance), models must still fallback on standard RLHF pipelines.

## 3. Prompting Rules & `<think>` Tokens (OpenAI o1/o3, DeepSeek-R1)
- **Eliminate CoT Triggers**: Do **NOT** use phrases like "Think step-by-step" or "Show your work." These models natively execute internal reasoning. Forcing external CoT conflicts with their RL-optimized thought vectors and degrades performance.
- **Minimal Few-Shot Examples**: Refrain from high-volume few-shot prompting. Pre-defining exact reasoning paths restricts the model’s autonomous logic exploration. Zero-shot or single-example prompts yield strictly better reasoning accuracy.
- **Constraint-Based Prompting**: State the "what" and the constraints, entirely omitting the "how." Use rigid delimiters (XML tags, Markdown) to separate context, data, and output goals instead of verbose procedural instructions.
- **Internal Monologue Integrity**: Do not constrain the style or persona of the reasoning phase. Attempting to force the model to "think like a persona" or "think in a specific language" corrupts the internal logic matrix and lowers the final output quality.
- **Parsing `<think>` Tokens**: DeepSeek R1 encapsulates raw reasoning inside `<think>...</think>` tags. System architectures must programmatically parse and strip these tags to isolate the final user-facing output from the raw reasoning trace.
