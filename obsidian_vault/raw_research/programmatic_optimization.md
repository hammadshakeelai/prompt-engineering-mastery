# Programmatic Prompt Optimization

## DSPy MIPROv2 (Multi-prompt Instruction Proposal Optimizer Version 2)
MIPROv2 is an optimizer in the DSPy framework designed to jointly optimize both instructions (system prompts) and few-shot demonstrations (examples).

### Key Features
- **Joint Optimization**: Evaluates and finds the optimal combination of instructions and examples simultaneously, avoiding suboptimal local minima found by tuning them in isolation.
- **Bayesian Optimization**: Uses Bayesian optimization to traverse the search space of prompts, balancing exploration of new combinations and exploitation of successful ones.
- **Bootstrapping**: Automatically bootstraps few-shot example candidates and grounds instructions in task-specific dynamics.

### API Reference
```python
dspy.MIPROv2(
    metric: Callable,
    prompt_model: Any | None = None,
    task_model: Any | None = None,
    teacher_settings: dict | None = None,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    auto: Literal['light', 'medium', 'heavy'] | None = 'light',
    num_candidates: int | None = None,
    num_threads: int | None = None,
    max_errors: int | None = None,
    seed: int = 9,
    init_temperature: float = 1.0,
    verbose: bool = False,
    track_stats: bool = True,
    log_dir: str | None = None,
    metric_threshold: float | None = None
)
```
- **Execution**: Optimized via `teleprompter.compile(program, trainset=trainset)`. The auto parameter (`'light'`, `'medium'`, `'heavy'`) determines the compute budget allocated for the search.

## DSPy v2.5
Version 2.5 marks the stabilization of several key features previously gated behind the `experimental` flag.

### Key Changes
- **Experimental Flag Removal**: Features like the Signature system, MIPROv2 optimizer, typed predictors, and the adapter system are now stable defaults. The `dspy.configure(experimental=True)` call is obsolete and should be removed.
- **Caching Controls**: The `dspy.LM` module introduced a `cache` parameter. Caching can now be explicitly disabled per instance: `dspy.LM('openai/gpt-4o-mini', cache=False)`.

## TextGrad (Textual Backpropagation)
TextGrad provides an automatic differentiation engine analogue for compound AI systems, substituting numerical gradients with natural language feedback ("textual gradients").

### Core Mechanism
- **Forward Pass**: System components (e.g., LLM calls) execute sequentially.
- **Loss Calculation**: An objective function or a "judge LLM" outputs a textual critique of the system's performance.
- **Backward Pass**: The critique propagates backward through the component chain. At each node, an LLM interprets the textual gradient and updates the internal state/prompt of that node to optimize performance.
- **Utility**: Allows parameter updating in non-differentiable systems, enabling optimization of prompts, code snippets, or agent reasoning steps using gradient descent methodologies applied to text.

## OPRO (Optimization by PROmpting)
Developed by Google DeepMind (arXiv:2309.03409), OPRO leverages Large Language Models as general-purpose optimizers without relying on gradient descent.

### Core Mechanism
- **Meta-prompting**: The LLM optimizer receives a meta-prompt containing a description of the optimization task and a history of previously generated solutions paired with their evaluation scores.
- **Iterative Refinement**: The LLM generates new candidate solutions (e.g., system prompts) based on historical performance data. These candidates are evaluated on the target task, and their scores are appended to the meta-prompt for the next optimization step.
- **Performance**: OPRO achieves up to an 8% improvement on the GSM8K benchmark and up to 50% on Big-Bench Hard (BBH) tasks over human-designed prompts.
