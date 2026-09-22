# Reflexion: Verbal Reinforcement Learning (Shinn et al. 2023)

Optimizes LLM agent behavior through verbal self-reflection rather than parameter updates:
1. **Actor**: Generates actions, reasoning traces, or trajectories (ReAct/CoT).
2. **Evaluator**: Assesses outputs against environment rewards, heuristics, or ground truth.
3. **Self-Reflection**: When trajectories fail, an LLM evaluates the error signal to generate constructive natural-language critiques.
Linguistic reflections are stored in episodic memory buffer and retrieved as context in subsequent trials to self-correct plans.