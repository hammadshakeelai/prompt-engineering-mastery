# 01. Core Principles of Prompt Engineering

Prompt engineering is the systematic discipline of designing, structuring, evaluating, and optimizing textual and multimodal inputs to steer Large Language Models (LLMs) toward deterministic, high-quality, and robust outputs.

---

## 1. Mechanics of LLM Generation

To engineer effective prompts, one must understand the autoregressive generative mechanism:

$$\mathbb{P}(w_1, w_2, \dots, w_T) = \prod_{t=1}^T \mathbb{P}(w_t \mid w_1, w_2, \dots, w_{t-1})$$

The model computes unnormalized log-odds (logits) over vocabulary $V$ given the token sequence context:

$$z_t = \text{Transformer}(\text{tokens}_{<t})$$
$$p(w_t = i) = \frac{\exp(z_{t, i} / \tau)}{\sum_{j=1}^{|V|} \exp(z_{t, j} / \tau)}$$

### Key Generation Hyperparameters and Their Prompt Engineering Impact

| Hyperparameter | Mechanism | Best Use in Prompting |
| :--- | :--- | :--- |
| **Temperature ($\tau$)** | Scales logit distribution sharpness. As $\tau \to 0$, distribution collapses to greedy decoding $\arg\max$. | Set $\tau \in [0.0, 0.2]$ for structured JSON, code, classification, reasoning. Set $\tau \in [0.7, 1.0]$ for creative writing, brainstorming. |
| **Top-$p$ (Nucleus)** | Truncates candidate pool to the smallest set of tokens whose cumulative probability $\ge p$. | Standard setting $p \in [0.85, 0.95]$. Avoid tuning both $\tau$ and $p$ simultaneously unless calibrating hallucination rates. |
| **Presence / Frequency Penalty** | Penalizes tokens based on presence in context or repetition frequency. | Use frequency penalties ($\sim 0.2 - 0.5$) when the LLM loops or gets stuck in repetitive lists. |
| **Stop Sequences** | Tokens that immediately halt generation when emitted. | Crucial for preventing model run-on, enforcing delimiter boundaries (e.g. `\nObservation:`, `</response>`). |

---

## 2. Foundational Pillars of Prompt Architecture

Every high-performance prompt follows an architectural separation of concerns.

```
+-----------------------------------------------------------------------+
| SYSTEM / DEVELOPER CONTEXT                                            |
| - Persona, Operational Scope, Security Envelope, Knowledge Boundary   |
+-----------------------------------------------------------------------+
| TASK DIRECTIVE                                                        |
| - Explicit, unambiguous imperative instruction                         |
+-----------------------------------------------------------------------+
| CONTEXT & REFERENCE MATERIAL (Delimited)                              |
| - Grounding facts, retrieved documents, schemas, environmental state   |
+-----------------------------------------------------------------------+
| CONSTRAINTS & NEGATIVE CONSTRAINTS                                    |
| - Invariants, forbidden operations, failure-handling modes            |
+-----------------------------------------------------------------------+
| DEMONSTRATIONS / FEW-SHOT EXEMPLARS                                   |
| - Input-Reasoning-Output tuples demonstrating edge cases              |
+-----------------------------------------------------------------------+
| INPUT PAYLOAD & TARGET TRIGGER                                        |
| - Target query wrapped in strict delimiters + output initiation cue   |
+-----------------------------------------------------------------------+
```

### 1. Persona & Identity vs Role Simulation
- **Shallow Role**: `"You are an expert python programmer."` (Weak prior, minimal impact on frontier models).
- **Behavioral Specification**: `"You are a Senior Systems Architect specializing in distributed consensus (Raft, Paxos). When evaluating designs, assess memory safety, network partitions under CAP theorem, and write amplification. Do not sugarcoat critiques."` (Strong prior, activates specific latent manifolds).

### 2. Delimiter Design & Structural Isolation
Frontier models are trained heavily on Markdown, XML, and JSON. Using explicit structural delimiters achieves two critical goals:
1. **Semantic segmentation**: Informs the attention mechanism where metadata ends and untrusted input begins.
2. **Injection barrier**: Mitigates indirect prompt injection by isolating external data payloads inside explicit tags.

Recommended Delimiters:
- **XML Tags** (Anthropic Claude native preference): `<context>`, `<user_input>`, `<instructions>`, `<scratchpad>`.
- **Markdown Headers**: `### Context`, `### Task Requirements`.
- **Triple Backticks or Quotes**: `"""` or ````json`.

### 3. Positive Framing vs Negative Constraints
LLMs attend to token representations. Negation tokens (`"Do not include comments"`) require the model to represent the forbidden concept ("comments") before suppressing it.
- **Sub-optimal**: `"Do not mention competitors. Do not write lengthy explanations. Do not use passive voice."`
- **Optimal (Prescriptive)**: `"Focus exclusively on our product's metrics. Output exactly two concise sentences. Write strictly in the active voice."`
- When negative constraints are mandatory, couple them with an explicit fallback or penalty:
  `"If the retrieved document does not contain the answer, output exactly: 'INFORMATION_NOT_FOUND'. Never speculate or synthesize outside facts."`

---

## 3. Context Window Dynamics & Positional Bias

Modern models support massive context windows (128k to 2M+ tokens), but attention distribution across sequence length is non-uniform.

### The "Lost in the Middle" Effect (Liu et al., 2023)
Performance follows a U-shaped curve:
- **Primacy Effect**: High recall for information in the first 5-10% of the prompt.
- **Recency Effect**: Highest recall for information at the very end of the prompt (last 5-10%).
- **Middle Degradation**: Information in the 40-70% depth range suffers up to 30-50% degradation in multi-document QA and complex reasoning.

### Strategic Layout Rules:
1. **Place System Instructions & Rules at the very top**.
2. **Place reference documents / long context in the middle**, tagged with semantic identifiers (`<doc id="doc_1">...</doc>`).
3. **Repeat critical constraints or provide the query and target trigger at the very bottom**.
   - Example: Place `<instruction_reminder>` right before the final `Human:` / `User:` payload.

---

## 4. The Tokenization Mental Model

LLMs process Byte-Pair Encodings (BPE) or SentencePiece tokens, **not** characters, syllables, or words.

### Common Tokenization Pitfalls:
1. **Character Counting & Spelling**:
   `"How many 'r's are in 'strawberry'?"` fails because `"strawberry"` is tokenized as `['str', 'aw', 'berry']`. The model does not see individual characters directly.
   - *Mitigation*: Prompt the model to space out characters or convert to code:
     `"First separate the letters with spaces ('s t r a w b e r r y'), then index and count them."`
2. **Numeric Arithmetic**:
   Numbers like `49219` might tokenize into `['49', '219']` or `['4', '921', '9']`, disrupting digit-by-digit algorithmic arithmetic.
   - *Mitigation*: Use scratchpads or write code (Code-as-Prompting / Tool use).
3. **Whitespace and Indentation**:
   Trailing spaces (e.g. `"Choose one: "` vs `"Choose one:"`) alter the prompt's final token representation, potentially shifting token probability distributions. Standardize trailing formatting across all prompts.

---

## 5. Production Prompt Template Standard

A hardened production prompt follows this standardized blueprint:

```markdown
<system_identity>
You are an enterprise data validation engine. You verify incoming vendor invoices against ISO-20022 compliance rules.
Operational Invariant: Never assume missing dates or tax IDs; reject invalid invoices immediately.
</system_identity>

<guidelines>
1. Evaluate each field against the target schema: [VendorName, TaxID, TotalAmount, Currency, LineItems].
2. Identify discrepancies, tax rate anomalies, or date mismatches.
3. If Currency is omitted, trigger "VALIDATION_FAILED: MISSING_CURRENCY".
</guidelines>

<output_format>
Return valid JSON matching this exact schema:
{
  "status": "APPROVED" | "REJECTED",
  "error_codes": string[],
  "normalized_data": object | null
}
Wrap your response strictly within ```json ... ``` blocks with no conversational preface or trailer.
</output_format>

<context_schema>
Currency rules: Only ["USD", "EUR", "GBP", "JPY"] permitted.
TaxID regex: "^[A-Z]{2}[0-9]{8,12}$"
</context_schema>

<input_payload>
Vendor: ACME Corp
TaxID: US99482103
Total: 4500
LineItems: [{ "item": "Cloud Compute", "qty": 10, "unit_price": 450 }]
</input_payload>

Analyze the payload and output the JSON result:
```
