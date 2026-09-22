import os

base = r'c:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\raw_research\snippets'
os.makedirs(base, exist_ok=True)

files = {
'emotion_prompting.md': """# EmotionPrompt: Psychological Stimulus Prompting (Cheng et al. 2023)

- **Mechanism**: Appends psychological stimuli derived from self-efficacy and emotional intelligence theories (e.g., *"This is very important to my career"*, *"Believe in your ability and exceed your limits"*).
- **Impact**: Yields ~8-11% relative accuracy improvements across BIG-bench and Instruction Induction benchmarks on ChatGPT and LLaMA-2.
- **Mechanistic Basis**: Emotional stimuli enrich representation quality and increase attention weights on key prompt tokens.""",

'plan_and_solve_prompting.md': """# Plan-and-Solve Prompting (Wang et al. 2023)

- **Two-Phase Cognitive Scaffolding**: Replaces generic Zero-Shot-CoT with explicit instruction: *"Let's first understand the problem and devise a plan... then carry out the plan..."*
- **Plan Generation**: Decomposes problems into an ordered sequence of sub-tasks without few-shot exemplars.
- **Plan Execution**: Sequentially carries out sub-tasks, preventing missing-step and calculation errors. PS+ adds variable extraction directives.""",

'test_time_compute_scaling.md': """# Test-Time Compute Scaling: Sequential vs. Parallel

- **Sequential Scaling (Longer CoT)**: Depth scaling where models dynamically plan, backtrack, and self-correct across extended reasoning traces; excels at multi-step tasks with tight dependency tracking.
- **Parallel Scaling (Best-of-N)**: Breadth scaling generating multiple independent candidate solutions evaluated via PRMs/ORMs or consensus voting; low wall-clock latency, high coverage.
- **Pareto Frontier**: Optimal test-time compute dynamically combines both sequential depth and parallel breadth.""",

'contrastive_cot_pairs.md': """# Contrastive Chain-of-Thought (Chia et al. 2023)

- **Contrastive Demonstrations**: Each exemplar pairs questions with both a negative (flawed/fallacious intermediate step) and a positive (valid step-by-step reasoning) trajectory.
- **Sharp Decision Boundaries**: Explicitly demonstrates common pitfalls and reasoning traps to avoid, substantially boosting accuracy on arithmetic and commonsense reasoning tasks."""
}

written = 0
for fname, content in files.items():
    path = os.path.join(base, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    written += 1

print(f'Wrote {written} snippet files.')
