# 01. DSPy Deep Dive: Compiling Declarative LM Pipelines

DSPy (Declarative Self-improving Python, Stanford NLP) replaces brittle, handcrafted string manipulation with a programming model that treats prompts as **compilable parameter weights**.

---

## 1. The Paradigm Shift: Strings vs. Signatures & Optimizers

```
Traditional Prompt Engineering:
Handcrafted String ---> Manual Tweaks ---> Model Update Breaks Prompt ---> Repeat 100x

DSPy Paradigm:
Declarative Signature ---> Metric Function ---> Teleprompter / Optimizer ---> Compiled SOTA Prompt
(Input/Output Spec)        (Success Criteria)   (Bayesian / Bootstrap Search)  (Auto-tuned for model)
```

Instead of hardcoding prompt strings, you define:
1. **Signatures**: Abstract specification of *what* the task takes and produces.
2. **Modules**: Architectural composition of execution (e.g. `Predict`, `ChainOfThought`, `ReAct`).
3. **Teleprompters (Optimizers)**: Algorithms that tune instructions and choose optimal few-shot exemplars to maximize a metric function.

---

## 2. Core DSPy Primitives

### A. Signatures
Signatures define the input/output contract:

```python
import dspy

class MultiHopFactChecker(dspy.Signature):
    """Verify claims by analyzing multiple retrieved evidentiary passages."""
    claim: str = dspy.InputField(desc="The factual claim to evaluate")
    context: list[str] = dspy.InputField(desc="List of retrieved search snippets")
    verdict: str = dspy.OutputField(desc="'SUPPORTED', 'REFUTED', or 'INSUFFICIENT'")
    rationale: str = dspy.OutputField(desc="Step-by-step logical reconciliation")
```

### B. Modules
Modules wrap signatures with cognitive strategies:
- `dspy.Predict(Signature)`: Direct single forward-pass.
- `dspy.ChainOfThought(Signature)`: Automatically injects a reasoning `rationale` field before outputs.
- `dspy.ReAct(Signature, tools=[...])`: Automatically constructs an autonomous ReAct loop.
- `dspy.ProgramOfThought(Signature)`: Prompts the model to express reasoning as executable Python code.

```python
class FactCheckingPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.reasoner = dspy.ChainOfThought(MultiHopFactChecker)
    
    def forward(self, claim):
        context = self.retrieve(claim).passages
        prediction = self.reasoner(claim=claim, context=context)
        return dspy.Prediction(verdict=prediction.verdict, rationale=prediction.rationale)
```

---

## 3. DSPy Optimizers (Teleprompters)

DSPy optimizers search the combinatorial space of prompt instructions and demonstration subsets.

```
+-------------------------------------------------------------------------------+
| Optimizer                           | Mechanism & When to Use                |
+-------------------------------------------------------------------------------+
| BootstrapFewShot                    | Generates complete (Input, CoT, Output) |
|                                     | traces using teacher model; filters    |
|                                     | traces that pass metric. Fast & zero cost.|
+-------------------------------------------------------------------------------+
| BootstrapFewShotWithRandomSearch    | Samples combinations of bootstrapped    |
|                                     | demos across multiple runs; picks best.|
+-------------------------------------------------------------------------------+
| MIPROv2 (Multi-prompt Instruction   | Proposes candidate instructions via LLM;|
| Proposal & Optimizer)               | uses Bayesian optimization to search   |
|                                     | joint space of instructions + demos.    |
|                                     | State of the Art for production systems.|
+-------------------------------------------------------------------------------+
```

### Compiling with MIPROv2:
```python
from dspy.teleprompt import MIPROv2

# 1. Define evaluation metric
def validate_verdict(gold, pred, trace=None):
    return gold.verdict.strip().upper() == pred.verdict.strip().upper()

# 2. Instantiate optimizer
teleprompter = MIPROv2(
    metric=validate_verdict,
    auto="medium",          # "light", "medium", or "heavy" compute budget
    num_candidates=10       # Instruction variations to synthesize
)

# 3. Compile: Searches instruction space and exemplar selection
compiled_pipeline = teleprompter.compile(
    FactCheckingPipeline(),
    trainset=train_dataset,
    valset=validation_dataset
)

# 4. Save compiled prompt artifacts
compiled_pipeline.save("compiled_fact_checker.json")
```

---

## 4. DSPy Assertions & Dynamic Backtracking

DSPy introduces compile-time and runtime assertions that trigger automated self-correction loops when outputs fail constraints:

```python
dspy.Assert(
    len(prediction.rationale) > 50,
    msg="Rationale is too terse. Expand on intermediate evidence.",
    target_module=self.reasoner
)
dspy.Suggest(
    prediction.verdict in ["SUPPORTED", "REFUTED", "INSUFFICIENT"],
    msg="Verdict must be one of the three enumerated states.",
    target_module=self.reasoner
)
```
When `dspy.Assert` fails, DSPy automatically rewinds execution, appends the error message to the prompt, and queries the model again up to $N$ retries.
