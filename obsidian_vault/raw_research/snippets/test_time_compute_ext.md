# Test-Time Compute Scaling and Reasoning Models

## 1. Test-Time Compute Scaling
Trades inference compute for accuracy by generating dynamic hidden Chain-of-Thought before final answers.
Models self-correct, explore hypotheses, and backtrack.
Scaled along two axes:
- Sequential: Longer CoT chains / thinking budgets.
- Parallel: Best-of-N, tree search / MCTS.
Performance scales log-linearly with inference compute on complex benchmarks.

## 2. Reinforcement Learning with Verifiable Rewards (RLVR)
Trains reasoning models using deterministic ground-truth verifiers (compilers, unit tests, symbolic math engines).
No subjective human preferences, no reward hacking, no labeling bottlenecks.
Policy autonomously discovers search strategies and extended reasoning trajectories.

## 3. DeepSeek-R1: Outcome-Based RL
Abandoned neural Process Reward Models (PRMs) due to:
- Cannot consistently define step boundaries for general tasks.
- High annotation cost and reward hacking vulnerability.
Uses rule-based outcome verification (accuracy + formatting) via Group Relative Policy Optimization (GRPO).
Proved pure outcome-based RL naturally elicits self-reflection and verification without step-level supervision.

## 4. Prompting Thinking Models
- Avoid "Think Step-by-Step": Models natively deliberate; manual CoT triggers degrade quality.
- Focus on Constraints and Intent: State objectives, format specs, and edge conditions plainly.
- Use Delimiters: XML tags or Markdown headers structure inputs and background documents.
- Provide Context, Not Solution Paths: Supply complete reference info; leave reasoning to the model.