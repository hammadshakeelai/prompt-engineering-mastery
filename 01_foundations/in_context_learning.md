# In-Context Learning (ICL) & Few-Shot Demonstration Engineering

In-Context Learning (ICL) is the emergent capability of autoregressive transformers to execute novel downstream tasks given only input-output examples in the context window, without updating any model parameters ($\Delta \theta = 0$).

---

## 1. Theoretical Underpinnings of ICL

How does an LLM learn in-context? Three prominent theoretical explanations:

1. **Implicit Gradient Descent (von Oswald et al., Dai et al.)**:
   The attention mechanism implements a form of implicit meta-optimization. Key-value projections act as parameter updates, computing an internal forward-backward gradient pass across the demonstrations.
2. **Bayesian Inference (Xie et al.)**:
   The prompt conditions the LLM to identify the underlying latent concept $q$ from a pre-trained mixture of tasks:
   $$p(y \mid x, \mathcal{D}_{\text{prompt}}) = \int p(y \mid x, \theta) p(\theta \mid \mathcal{D}_{\text{prompt}}) d\theta$$
3. **Task Retrieval & Induction Heads (Olsson et al., Anthropic)**:
   Specific two-layer attention head circuits ("induction heads") recognize repeated patterns of the form `[A][B] ... [A] -> [B]`, copying and transforming demonstrations dynamically.

---

## 2. Inherent Biases in In-Context Learning

Zhao et al. (2021) identified three catastrophic biases that undermine naive few-shot prompts:

```
[Naive Demonstrations] ---> [Majority Label Bias]  (Favors frequent demo labels)
                        ---> [Recency Bias]         (Favors label of final demo)
                        ---> [Common Token Bias]    (Favors frequent pre-training tokens)
```

1. **Majority Label Bias**: If 3 out of 4 examples in the prompt are labeled `POSITIVE`, the model skews towards predicting `POSITIVE` regardless of input semantics.
2. **Recency Bias**: The model disproportionately outputs the class label that appeared in the very last exemplar.
3. **Common Token Bias**: Words frequently encountered during pretraining (e.g. "good", "company") are favored over domain-specific tokens (e.g. "parsimonious", "refactor").

### Mitigating Bias: Contextual Calibration
To calibrate, evaluate the prompt on a "content-free" input (e.g., `Input: N/A\nOutput:` or `Input: [MASK]\nOutput:`).
If $P(\text{Positive} \mid \text{"N/A"}) = 0.8$ and $P(\text{Negative} \mid \text{"N/A"}) = 0.2$, fit an affine transformation $W, b$ on output logits such that:
$$\hat{p} = \text{Softmax}(W \cdot \text{logits} + b)$$
where $W$ and $b$ force the prior on empty input to be uniform.

---

## 3. Exemplar Selection Strategies

Selecting the right demonstrations is the single highest-leverage factor in ICL performance.

```
Incoming Query (x)
       |
       +---> [1. Dense Embedding Retrieval] (k-NN via cosine similarity)
       |
       +---> [2. Maximal Marginal Relevance] (Balance similarity with diversity)
       |
       +---> [3. Hard-Negative / Boundary Selection] (Exemplars near decision boundary)
```

### A. Dynamic k-NN Retrieval (RAG for Few-Shot)
Instead of static hard-coded examples, query a vector database (e.g., Chroma, FAISS, pgvector) of 1,000+ curated domain examples:

```python
def select_exemplars_knn(query_vector, exemplar_store, k=4):
    """
    Retrieve top-k exemplars maximizing cosine similarity to target input.
    """
    similarities = cosine_similarity(query_vector, exemplar_store.vectors)
    top_indices = np.argsort(similarities)[-k:]
    return [exemplar_store.items[i] for i in reversed(top_indices)]
```

### B. Maximal Marginal Relevance (MMR)
Prevent retrieving 4 virtually identical examples. MMR penalizes redundant exemplars:

$$\text{MMR}(Q, D) = \arg\max_{d_i \in R \setminus S} \left[ \lambda \text{Sim}_1(d_i, Q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}_2(d_i, d_j) \right]$$

- $\lambda = 0.7$: Prioritize relevance while ensuring structural/syntactic diversity in examples.

---

## 4. What Matters in Demonstrations? (Min et al., 2022)

Groundbreaking empirical studies revealed surprising truths about what LLMs learn from few-shot examples:

| Element | Impact on Classification / Extraction | Impact on Multi-Step Reasoning |
| :--- | :--- | :--- |
| **Input Format & Syntax** | **CRITICAL** (Teaches token schema, casing, delimiter structure) | **CRITICAL** |
| **Input Distribution** | **HIGH** (Informs domain vocabulary and feature space) | **HIGH** |
| **Label Space** | **HIGH** (Restricts output to valid enum keys) | **HIGH** |
| **Ground-Truth Label Correctness** | **LOW to MODERATE** (Randomized labels drop accuracy by only 5-10% in simple classification!) | **FATAL** (In multi-step reasoning, flawed intermediate logic degrades output by 60%+) |

> [!IMPORTANT]
> For complex reasoning (math, code, architecture), **reasoning traces must be 100% verified and rigorous**.
> For simple classification, consistency of formatting and enumeration of valid classes matters more than label accuracy.

---

## 5. Exemplar Structuring Blueprint

When constructing few-shot exemplars, follow the **Context-Input-Thought-Output (CITO)** schema:

```markdown
<example id="ex_1">
<input>
Analyze code snippet for concurrency vulnerabilities:
```python
def deposit(account_id, amount):
    balance = get_balance(account_id)
    set_balance(account_id, balance + amount)
```
</input>
<reasoning>
1. Thread A and Thread B concurrently call deposit for account_id=1 with amount=100.
2. Thread A reads balance = 500.
3. Thread B reads balance = 500 before Thread A commits.
4. Thread A writes balance = 600.
5. Thread B overwrites balance = 600.
6. True expected balance is 700. This is a classic Time-of-Check to Time-of-Use (TOCTOU) race condition.
</reasoning>
<output>
{
  "vulnerability": "RACE_CONDITION",
  "cwe": "CWE-362",
  "severity": "HIGH",
  "recommendation": "Use atomic database increment operations (UPDATE ... SET balance = balance + :amt) or acquire a distributed mutex (e.g. Redis Redlock)."
}
</output>
</example>
```
