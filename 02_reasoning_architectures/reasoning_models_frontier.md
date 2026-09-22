# 03. Prompting Frontier Reasoning Models (OpenAI o1/o3, DeepSeek-R1, Gemini Thinking)

The release of models optimized via Test-Time Compute Scaling and Reinforcement Learning with Verifiable Rewards (RLVR)—notably **OpenAI o1/o3**, **DeepSeek-R1**, and **Google Gemini 2.0 Thinking**—fundamentally upends classical prompt engineering conventions.

---

## 1. Test-Time Compute & Hidden Reasoning Mechanisms

In standard models (GPT-4o, Claude 3.5 Sonnet), every generated token consumes the same fixed floating-point operations (FLOPs).
Reasoning models decouple input token length from reasoning token allocation:

$$\text{Tokens}_{\text{total}} = \text{Tokens}_{\text{reasoning}} (hidden) + \text{Tokens}_{\text{completion}} (visible)$$

Through large-scale Reinforcement Learning (RL), the model has learned to:
- Formulate internal hypotheses
- Verify edge cases before generating visible text
- Recognize flawed assumptions and explicitly backtrack
- Double-check arithmetic and structural constraints

```
User Prompt ---> [Hidden Internal CoT: 2,000 - 32,000 tokens] ---> Final Answer
                 - Self-correction
                 - Sub-goal decomposition
                 - Boundary stress testing
```

---

## 2. Why Classical Prompt Engineering Degrades Reasoning Models

Techniques developed for non-reasoning LLMs often **actively harm** reasoning models.

| Classical Technique | Effect on Standard Models | Effect on Reasoning Models (o1/R1) | Root Cause |
| :--- | :--- | :--- | :--- |
| **"Think step by step"** | Dramatic boost (+20-30%) | **Degrades or neutral** | Redundant. Competes with or interrupts native RL policy search. |
| **Prescriptive Few-Shot CoT** | High improvement | **Degrades (-5% to -15%)** | Forces the model to emulate a rigid human reasoning structure rather than exploring its own optimal search paths. |
| **Over-specified Steps** | Improves adherence | **Constrains exploration** | Micro-managing intermediate steps prevents the model from discovering superior verification routes. |
| **Persona Prompting ("You are Einstein")** | Slight stylistic framing | **Distracts latent focus** | Adds irrelevant token priors into the internal reasoning search tree. |

---

## 3. The New Rules of Prompting Reasoning Models

### Rule 1: Specify the *What*, Not the *How* (Outcome-Oriented Prompting)
Define explicit goals, constraints, invariants, and evaluation rubrics. Allow the internal reasoning mechanism complete freedom over intermediate computations.

### Rule 2: Provide Rich, Raw Context (Don't Pre-digest)
Because these models possess superior attention navigation and verification abilities, you do not need to summarize or extract data beforehand. Feed raw documentation, logs, or codebases directly.

### Rule 3: Enforce Crisp Delimiters for Final Output
Instruct the model to encapsulate its final answer cleanly so downstream orchestrators can bypass internal reasoning residue:
```markdown
Provide your final verdict strictly within <final_answer>...</final_answer> tags.
```

### Rule 4: Leverage Structured Edge-Case Boundary Conditions
List precise failure boundaries the model must safeguard against. The model will internally construct adversarial verification tests against your constraints.

---

## 4. Side-by-Side Prompt Comparison

### Anti-Pattern (Classical Style Applied to o1/R1)
```markdown
You are a brilliant Senior Database Engineer. Let's think step by step.
First, look at the table schema below.
Second, analyze whether table indexing is optimal.
Third, explain step 1, step 2, step 3 of why a hash index is worse than a B-tree index here.
Finally, give me the SQL migration script.
```
*Why it fails*: Micro-manages the steps, forces redundant explanations, restricts internal search.

### State-of-the-Art Pattern (Optimized for o1/o3/DeepSeek-R1)
```markdown
<objective>
Generate a production PostgreSQL migration script to optimize query performance for an append-only event log table with 200M rows.
</objective>

<workload_profile>
- 98% writes: High-throughput batch inserts (10,000 events/sec).
- 2% reads: Time-range queries strictly bounded by (tenant_id, created_at DESC).
- Hardware: AWS RDS Aurora PostgreSQL, 32 vCPU, 128 GB RAM.
</workload_profile>

<hard_constraints>
1. Migration must run online without taking table-exclusive lock (SHARE UPDATE EXCLUSIVE only).
2. Write throughput cannot degrade by more than 5% during index creation.
3. Storage overhead for indices must not exceed 25% of table size.
4. Account for partition maintenance if range partitioning is selected.
</hard_constraints>

<output_requirements>
Encapsulate the runnable SQL migration within ```sql ... ```.
Follow with a concise verification matrix detailing how each hard constraint is satisfied.
</output_requirements>
```

---

## 5. Controlling Reasoning Effort and Test-Time Compute

Frontier APIs allow tuning the reasoning budget:

```python
# OpenAI API (o1 / o3-mini)
response = client.chat.completions.create(
    model="o3-mini",
    messages=[{"role": "user", "content": prompt}],
    reasoning_effort="high"  # Options: low, medium, high
)

# DeepSeek-R1 / Gemini 2.0 Flash Thinking
# DeepSeek returns reasoning tokens inside the <think> ... </think> tag,
# allowing developers to inspect or strip reasoning traces programmatically.
```
- **`low`**: For fast syntax transformation, refactoring, and structured JSON parsing.
- **`medium`**: Standard debugging, architectural reviews, and algorithm synthesis.
- **`high`**: Mathematical proofs, cryptography validation, concurrency race condition analysis, and zero-day vulnerability audits.
