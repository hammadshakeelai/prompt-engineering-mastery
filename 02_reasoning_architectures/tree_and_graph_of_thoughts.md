# Tree of Thoughts (ToT) & Graph of Thoughts (GoT)

Traditional Chain-of-Thought (CoT) enforces a strictly linear trajectory of token exploration. When a model commits a logical fallacy at step 2, autoregressive generation carries that error through subsequent steps without the capacity to backtrack.

Tree of Thoughts (ToT) and Graph of Thoughts (GoT) generalize prompting into non-linear, state-space search algorithms.

---

## 1. Tree of Thoughts (ToT) Architecture (Yao et al., 2023)

ToT treats problem solving as search over a tree where each node represents a partial solution ("thought") $s = [x, z_1, \dots, z_i]$.

```
                     [Root: Initial Problem x]
                         /          \
                     [Step 1a]      [Step 1b]
                     /      \           \
                 [Step 2a] [Step 2b]   [Step 2c]  <-- Backtrack & Prune invalid branches
                    |
                 [Final Solution y]
```

### The Four Modules of ToT

1. **Thought Decomposer**: Breaks problem down into discrete thought units (e.g., a mathematical sub-equation, a paragraph outline, a chess move candidate).
2. **Thought Generator ($G$)**:
   - *Sample ($G_{\text{sample}}$)*: Generate $k$ independent candidate thoughts i.i.d. via high temperature ($\tau=0.7$).
   - *Propose ($G_{\text{propose}}$)*: Sequentially propose distinct candidate moves via a single prompt.
3. **State Evaluator ($V$)**:
   - *Value ($V_{\text{value}}$)*: Assign a scalar value $v \in [0, 1]$ or categorical rating (`sure`, `maybe`, `impossible`) to a partial thought state.
   - *Vote ($V_{\text{vote}}$)*: Compare multiple candidates across branches and select the most promising.
4. **Search Algorithm**:
   - **Breadth-First Search (BFS)**: Maintain the top-$b$ most promising states at each depth (Beam Search).
   - **Depth-First Search (DFS)**: Explore paths until reaching leaf or threshold, with programmatic backtracking when value $v < \text{threshold}$.

---

## 2. ToT Prompt Implementation Patterns

### Pattern 1: State Evaluator / Value Function Prompt
```markdown
<system>
You are an algorithmic validator evaluating partial states in a constraint optimization task.
</system>

<context>
Target: Make the number 24 using numbers [4, 7, 8, 8] with basic operations (+, -, *, /).
Current partial state:
- Used: 8 / (4 - ?)
Remaining numbers: [7, 8]
</context>

<evaluation_task>
Evaluate whether the current partial state can reach 24.
Analyze possible next steps. Then output:
EVALUATION: [SURE | MAYBE | IMPOSSIBLE]
SCORE: [0.0 to 1.0]
REASONING: Concise 1-sentence justification.
</evaluation_task>
```

---

## 3. Graph of Thoughts (GoT) (Besta et al., ETH Zurich, 2023)

Graph of Thoughts expands trees into arbitrary Directed Acyclic Graphs (DAGs), enabling operations impossible in trees:

```
    Node A (Draft 1) -------\
                             +---> Node C (Synthesize & Reconcile) ---> Node E (Final Polish)
    Node B (Draft 2) -------/           ^
                                        |
    Node D (Adversarial Critique) ------+
```

### Graph Operations Supported:
1. **Branching (1-to-N)**: Forking multiple hypotheses from a single premise.
2. **Aggregation (N-to-1)**: Merging the strengths of multiple distinct thought branches into a unified superior synthesis.
3. **Refinement (Feedback Loop)**: Node $C$ inspects Node $A$ against critique Node $B$ and iteratively updates.

---

## 4. Algorithm of Thoughts (AoT) (Sel et al., 2023)

ToT requires dozens to hundreds of separate API calls, resulting in high latency and token cost.
**Algorithm of Thoughts (AoT)** compresses tree-search exploration (generation, evaluation, backtracking) directly into a **single continuous context stream** by prompting the model to emulate DFS/BFS algorithms in-context:

```markdown
<system>
You solve problems by systematically exploring a search tree in a single pass.
For each branch:
1. Propose candidate sub-step.
2. Evaluate feasibility: If dead-end, explicitly output "[BACKTRACK to step N]" and explore alternative.
3. Mark confirmed branches with "[VALIDATED]".
</system>

Problem: Design an idempotent payment processing webhook endpoint.
```
