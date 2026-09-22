import os

base = r'c:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\raw_research\snippets'
os.makedirs(base, exist_ok=True)

files = {
'textgrad.md': """# TextGrad: Automatic Differentiation via Text (Yuksekgonul et al., 2024)

- **Computation Graph Optimization**: Represents multi-component LLM workflows as PyTorch-like computation graphs where nodes represent variables (prompts, code, intermediate reasoning) and edges represent LLM calls. Prompts are tunable parameters.
- **LLM Backpropagation**: LLMs act as critics evaluating outputs against loss functions, generating structured qualitative critiques termed *textual gradients*. These gradients are propagated backward in reverse topological order.
- **Automatic Differentiation**: Aggregates textual gradients at target prompt variables; an LLM optimizer updates prompt text iteratively without weight access.""",

'least_to_most.md': """# Least-to-Most Prompting (Zhou et al., 2022)

1. **Sub-problem Decomposition**: Prompts direct the model to break down a complex target problem into an ordered list of simpler, intermediate subproblems.
2. **Sequential Solving with Accumulated Context**: Subproblems are solved sequentially in ascending complexity. Each question and generated answer are appended to context for subsequent steps.
3. **Compositionality Generalization**: Enables solving problems requiring more steps than seen in exemplars (e.g., 99.7% on SCAN length splits vs. 16.2% standard CoT).""",

'opro.md': """# OPRO: Optimization by PROmpting (Yang et al., DeepMind 2023)

- **LLM as Optimizer**: Derivative-free, black-box optimization defined entirely in natural language.
- **Meta-Prompting & Trajectory**: Guided by problem description + optimization trajectory (sorted past candidate solutions with scores). LLM recognizes trends and balances exploration/exploitation.
- **Results**: On GSM8K, boosted accuracy up to 8% (reaching 80.2% on PaLM 2-L with prompts like *"Take a deep breath and work on this problem step-by-step"*); up to 50% relative gain across BBH tasks.""",

'skeleton_of_thought.md': """# Skeleton-of-Thought (SoT) (Ning et al., ICLR 2024)

- **Skeleton Drafting**: Generates a concise outline/bullet points instead of full sequential text.
- **Point-by-Point Parallel Generation**: Each skeleton point is elaborated simultaneously and independently via concurrent API requests or batched GPU decoding.
- **Speedup**: Up to 2.39x average speedup across question types; peak acceleration 2.69x-2.88x without quality loss.
- **Latency**: Exploits GPU batch parallelization during decoding to cut memory-bound token latency.""",

'thread_of_thought.md': """# Thread of Thought (ThoT) (Zhou et al., 2023)

- **Incremental Segment Summary**: Prompts model to divide context into manageable chunks, analyzing and summarizing sequentially before final synthesis.
- **Chaotic Context Handling**: Mitigates "lost-in-the-middle" and distractor corruption in RAG/multi-turn dialogue by maintaining balanced attention across all document sections.
- **Vs. CoT**: CoT focuses on logical deduction; ThoT acts as a cognitive digest filter prioritizing evidence extraction in noisy contexts.""",

'promptbreeder.md': """# Promptbreeder: Self-Referential Prompt Evolution (Fernando et al., 2023)

- **Co-Evolution**: Evolves task-prompts coupled with meta-level mutation-prompts simultaneously. Mutation-prompts undergo "hypermutation," refining the optimizer itself.
- **Evolutionary Strategies**: Population-based genetic algorithm driven by binary tournament selection, mutation operators, and estimation-of-distribution algorithms.
- **Fitness Landscapes**: Traverses rugged, non-differentiable discrete prompt spaces by dynamically adapting mutation operators to landscape geometry.""",

'step_back_prompting.md': """# Step-Back Prompting (Zheng et al., 2023)

1. **Abstraction**: Model generates and answers a broader, higher-level "step-back question" stripping away instance-specific noise.
2. **Reasoning**: Solves original question grounded in the retrieved first principles/governing concepts.
3. **Benchmarks**: +7% MMLU Physics, +11% MMLU Chemistry, up to +36% on TimeQA/MuSiQue across PaLM-2L, GPT-4, and LLaMA-2-70B.""",

'graph_of_thoughts.md': """# Graph of Thoughts (GoT) (Besta et al., 2023)

- **Graph Topology**: Models reasoning as arbitrary directed graphs $(G, T, E, R)$ where vertices are thoughts and edges capture dependencies.
- **Node Aggregation**: Allows multiple incoming edges to synthesize, reconcile, and combine independent reasoning paths.
- **Backtracking & Cycles**: Flexible network-level backtracking and recurrent feedback loops without restarting entire subtrees.
- **Vs. ToT**: Non-tree structures yield +62% sorting accuracy while cutting inference costs >31% via thought reuse.""",

'self_consistency.md': """# Self-Consistency in CoT (Wang et al., 2022)

- **Mechanism**: Samples diverse reasoning paths via temperature sampling ($T=0.5-0.7$) and takes unweighted majority vote over answers.
- **Sample Size**: Steepest gains occur at $k=5-10$; marginal returns diminish toward $k=40$.
- **Calibration**: Consensus vote share acts as an unsupervised, well-calibrated confidence score.
- **Benchmarks**: +10.6% to +17.9% GSM8K, +11% to +14% SVAMP, up to +24% MultiArith (reaching 99.3%)."""
}

written = 0
for fname, content in files.items():
    path = os.path.join(base, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    written += 1

print(f'Wrote {written} snippet files.')
