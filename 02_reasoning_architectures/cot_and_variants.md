# 02. Chain of Thought (CoT) & Advanced Reasoning Variants

Chain of Thought (CoT) prompting fundamentally alters transformer generation by dedicating autoregressive computation (intermediate tokens) to intermediate logical steps before generating the final answer.

---

## 1. Why CoT Works: The Computational View

Standard autoregressive generation computes:
$$P(Y \mid X) = \prod_{t=1}^M P(y_t \mid X, y_{<t})$$
If $Y$ requires complex reasoning, the model has only a constant number of transformer layers to compute the answer in a single forward pass.

With Chain of Thought:
$$P(Y, Z \mid X) = \left[ \prod_{k=1}^K P(z_k \mid X, z_{<k}) \right] \times \left[ \prod_{t=1}^M P(y_t \mid X, Z, y_{<t}) \right]$$
Where $Z = (z_1, \dots, z_K)$ represents intermediate reasoning tokens.
- **Dynamic Compute Scaling**: Every generated token $z_k$ triggers an additional forward pass through the entire transformer stack ($L$ layers).
- **Working Memory**: Previous tokens stored in the KV-cache serve as an external, readable working memory.

---

## 2. Spectrum of CoT Methodologies

```
+--------------------------------------------------------------------------+
| Reasoning Prompt Methodologies                                           |
+--------------------------------------------------------------------------+
| 1. Zero-Shot CoT        | "Let's think step by step." (Kojima et al.)    |
| 2. Few-Shot CoT         | Multi-step exemplars (Wei et al.)              |
| 3. Self-Consistency     | Sample N paths at τ > 0; majority vote answer  |
| 4. Least-to-Most        | Decompose into subproblems; solve sequentially |
| 5. Step-Back Prompting  | Derive high-level principle before solving     |
| 6. Skeleton-of-Thought  | Plan skeleton first, expand points in parallel |
+--------------------------------------------------------------------------+
```

---

## 3. Deep-Dive: Variants & Templates

### A. Zero-Shot CoT (Kojima et al., 2022)
Adding a simple cognitive trigger activates reasoning pathways.
- Classic trigger: `"Let's think step by step."`
- Modern calibrated triggers:
  - Plan-and-Solve: `"Let's first understand the problem, extract relevant variables, devise a plan, and carry out the steps."`
  - Critical Review: `"Take a deep breath and work through this systematically, verifying each deduction."`

### B. Self-Consistency (Wang et al., 2022)
Human reasoning relies on multiple perspectives. Instead of greedy decoding ($\tau=0$), Self-Consistency samples $N$ distinct reasoning trajectories with $\tau \in [0.5, 0.7]$ and applies marginalization / majority voting over the final answers:

$$\hat{y} = \arg\max_y \sum_{i=1}^N \mathbf{1}(y^{(i)} = y)$$

```
                    +---> Path 1 (Reasoning A) ---> Answer: $42
                    |
Prompt (Input X) ---+---> Path 2 (Reasoning B) ---> Answer: $42   ===> WINNER: $42
(Sample at τ=0.7)   |                                                 (Confidence: 66%)
                    +---> Path 3 (Reasoning C) ---> Answer: $55
```

*Tradeoff*: Increases token consumption and latency by factor of $N$, but yields dramatic improvements (+10% to +25%) in GSM8K, SVAMP, and logic benchmarks.

### C. Least-to-Most Prompting (Zhou et al., 2022)
Solves problems that exceed the model's single-pass context horizon by:
1. **Decomposition Phase**: Prompt the model to break the complex problem $Q$ into subproblems $[q_1, q_2, \dots, q_k]$.
2. **Sequential Resolution Phase**:
   - Solve $q_1 \to a_1$.
   - Pass $(q_1, a_1)$ as context to solve $q_2 \to a_2$.
   - Iterate until reaching $q_k \to A_{\text{final}}$.

```markdown
<decomposition_prompt>
Target Problem: What is the total energy consumption of a data center running 500 nodes with 80% CPU utilization and 200 nodes idle over 24 hours?

Break this problem down into atomic, prerequisite questions. Do not solve yet.
</decomposition_prompt>
```

### D. Step-Back Prompting (Zheng et al., Google DeepMind, 2023)
LLMs often hallucinate when diving directly into domain details. Step-Back prompting forces abstraction before execution:
1. **Abstraction**: Ask a higher-level question about the governing physics, mathematical principle, or business logic.
2. **Grounding**: Answer the original specific question using the derived abstraction.

```markdown
<system>
You are an expert distributed systems engineer.
</system>

<step_back_instruction>
Before answering the specific question, identify and state the underlying distributed systems theorem or architectural constraint (e.g., PACELC, Amdahl's Law, Little's Law) that governs this scenario.
</step_back_instruction>

<user_query>
If we add 4 more read replicas to our primary PostgreSQL database with synchronous replication, will our write latency decrease?
</user_query>
```
*Model response abstraction*: "First, under synchronous replication, write latency is bounded by the slowest replica acknowledging the transaction (two-phase commit overhead). Little's Law and Amdahl's Law apply..."

---

## 4. Directional Stimulus & Skeleton-of-Thought

### Directional Stimulus Prompting (Li et al., 2023)
A small, auxiliary model (or lightweight classifier) generates high-level "hints" or "directional stimuli" (e.g. keywords, key lemmas) injected into the prompt of the primary LLM to steer reasoning without handcrafting.

### Skeleton-of-Thought (SoT) (Ning et al., 2023)
Reduces latency by up to 2.5x:
1. **Skeleton Phase**: The LLM outputs a concise numbered outline of points.
2. **Parallel Expansion Phase**: The orchestrator spawns concurrent LLM calls, each expanding one bullet point in parallel.
3. **Reassembly**: Concatenates expanded points into the finished report.
