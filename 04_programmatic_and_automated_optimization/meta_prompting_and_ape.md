# 02. Meta-Prompting & Automatic Prompt Engineering (APE)

Meta-Prompting leverages LLMs as meta-optimizers to programmatically generate, critique, mutate, and calibrate prompts without human intervention.

---

## 1. Automatic Prompt Engineer (APE) (Zhou et al., 2022)

APE formalizes prompt engineering as a natural language optimization problem:

$$\rho^* = \arg\max_{\rho} \mathbb{E}_{(x, y) \sim \mathcal{D}} [f(\mathcal{M}(\rho, x), y)]$$

Where:
- $\rho$ is the candidate prompt instruction.
- $\mathcal{M}(\rho, x)$ is the language model output given prompt $\rho$ and input $x$.
- $f$ is a scoring function (e.g., 0-1 exact match, log-likelihood, semantic similarity).

```
[Dataset: Inputs (x) & Outputs (y)]
                 |
                 v
   +------------------------------+
   | 1. Candidate Proposal (LLM)  | ---> Generates candidate instructions {ρ_1, ρ_2, ... ρ_N}
   +------------------------------+
                 |
                 v
   +------------------------------+
   | 2. Candidate Evaluation      | ---> Computes performance metric f on validation batch
   +------------------------------+
                 |
                 v
   +------------------------------+
   | 3. Iterative Resampling /    | ---> Mutates top-performing candidates; generates variations
   |    Monte Carlo Optimization  |
   +------------------------------+
                 |
                 v
           [Optimal Prompt ρ*]
```

### The Proposal Prompt (LLM as Prompt Generator):
```markdown
<system>
You are an expert prompt engineer. You discover prompts that maximize downstream model performance.
</system>

<task_dataset>
Input: "The battery lasts only 2 hours." -> Output: "BATTERY_DEFECT"
Input: "Screen flickers when brightness is 100%." -> Output: "DISPLAY_GLITCH"
Input: "Delivered 4 days after promised arrival." -> Output: "LOGISTICS_DELAY"
</task_dataset>

<meta_instruction>
Analyze the input-output pairs above. Formulate 8 diverse candidate system prompts that instruct an LLM to perform this classification task with 100% precision. Make instructions explicit, concise, and unambiguous.
</meta_instruction>
```

---

## 2. Evolutionary Prompt Optimization (PromptBreeder, DeepMind)

Fernando et al. (2023) demonstrated that self-referential evolutionary algorithms evolve both the **task prompt** and the **mutation prompts** over successive generations.

```
Generation g:
[Prompt Pop P_1 ... P_K] ---> [Evaluate Fitness] ---> [Select Top Performers]
                                                              |
                                                              v
[Mutated Pop g+1] <--- [Crossover & Semantic Mutation Operators]
```

### Evolutionary Operators:
1. **Instruction Crossover**: Combining the behavioral persona of Candidate $A$ with the output constraint format of Candidate $B$.
2. **First-Principles Mutation**: Asking the LLM to rewrite the prompt from a completely different pedagogical perspective (e.g., Socratic method, mathematical formalism).
3. **Hypermutation**: Mutating the prompt that mutates the prompts!

---

## 3. Anthropic Meta-Prompt Architecture

Anthropic's frontier prompt generator follows a systematic multi-turn meta-prompt template:

1. **Input Task Analysis**: Extracts core variables `{$VAR1}`, `{$VAR2}` from the user request.
2. **Mental Sandbox Simulation**: Simulates edge-case failures the model might encounter.
3. **Drafting Instructions**: Enforces XML structuring, strict delimiters, and scratchpad reasoning.
4. **Few-Shot Synthesis**: Programmatically generates 3 canonical examples containing positive and negative test cases.

```markdown
<meta_prompt_template>
I need a prompt designed for: {USER_GOAL}

Follow these steps:
1. Deconstruct the requirements into: Objective, Input Variables, Constraints, and Output Format.
2. Identify 3 potential edge cases or failure modes.
3. Write a production-ready prompt using XML tags (<system_instructions>, <context>, <rules>, <output_schema>).
4. Include a <scratchpad> instruction so the model plans before answering.
</meta_prompt_template>
```
