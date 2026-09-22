---
name: evolutionary-prompt-optimizer
description: Specialized directive for automated evolutionary meta-prompt optimization (PromptBreeder, OPRO, AutoPDL, TextGrad, DSPy), self-referential mutation operators, fitness landscape navigation, and programmatic prompt compilation.
---

# Evolutionary Prompt Optimizer Skill

Use this skill when automating prompt engineering, compiling optimal prompts for complex agent pipelines, running self-improving prompt evolution (PromptBreeder, OPRO, TextGrad), or navigating complex discrete prompt search spaces with programmatic objective functions.

## 1. Evolutionary Meta-Prompting (PromptBreeder Protocol)

1. **Dual-Population Dynamics:**
   - Maintain two co-evolving populations:
     - **Task Prompts ($\mathcal{P}_{\text{task}}$):** Concrete system instructions executed by the target language model on downstream tasks.
     - **Mutation Prompts ($\mathcal{M}$):** Self-referential meta-prompts that dictate *how* task prompts are modified, recombined, and evolved over generations.
   - At each generation $g$, evaluate task prompt fitness $f(p) = \frac{1}{|D_{\text{val}}|} \sum_{(x, y)} \mathcal{S}(p(x), y)$.
   - Apply mutation prompts to top-performing task prompts to generate offspring: $p' \sim \text{LLM}(m(p))$.

2. **Self-Referential Hyper-Mutation:**
   - Do not fix mutation operators statically. Allow mutation prompts $\mathcal{M}$ to mutate themselves via higher-order meta-mutation operators:
     $$m' \sim \text{LLM}(\text{MetaMutation}(m))$$
   - This discovers domain-specific optimization heuristics (e.g., adversarial prompt hardening, dynamic zero-shot thinking tags, formatting distillation).

## 2. Textual Gradient Optimization (TextGrad Protocol)

1. **Computational Graph of Text:**
   - Model the prompt evaluation pipeline as a directed acyclic graph (DAG) where nodes represent text variables (system prompt, input, intermediate CoT reasoning, output) and edges represent LLM calls or objective functions.
2. **Textual Loss & Backpropagation:**
   - Compute textual feedback (the "gradient") at the output node using an evaluation prompt:
     $$g_{\text{output}} = \nabla_{\text{text}} \mathcal{L}(\text{output}, \text{target})$$
   - Backpropagate feedback through intermediate reasoning steps to the upstream system prompt node.
   - Update prompt $p$ via a textual optimizer (e.g., `TGD` / Textual Gradient Descent with momentum):
     $$p^{(t+1)} = \operatorname{Optimizer}(p^{(t)}, g_p^{(t)})$$

## 3. Programmatic Prompt Compilation (DSPy Framework)

1. **Modular Signature Separation:**
   - Decouple pipeline logic from textual prompt phrasing. Define tasks via declarative signatures (`Inputs -> Outputs`) and modular components (`dspy.Predict`, `dspy.ChainOfThought`).
2. **Automated Few-Shot Demonstration Mining:**
   - Use `BootstrapFewShotWithRandomSearch` or `MIPROv2` (Multi-prompt Instruction Proposal and Optimizer).
   - Jointly optimize both:
     - The meta-instruction string (task prompt).
     - The curated set of exemplar demonstrations (selected dynamically from training traces passing strict metric thresholds).
3. **Pareto Optimization:**
   - Evaluate multi-objective fitness: task accuracy, token latency, output format adherence, and hallucination penalty. Maintain a Pareto frontier of candidate prompt architectures.
