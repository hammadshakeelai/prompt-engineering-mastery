# 02. Structured Outputs & Tool Calling Architecture

Unstructured text outputs are brittle for production software integration. Reliable systems demand deterministic schemas, constrained grammar decoding, and hardened tool-calling interfaces.

---

## 1. The Spectrum of Structured Generation

```
[Prompt Guidance Only]       [Self-Correction Loop]      [Grammar-Constrained Decoding]
("Output valid JSON...")     (Pydantic validate & retry) (Outlines / LMQL / OpenAI Strict)
---------------------------------------------------------------------------------------->
Determinism: Low             Determinism: High            Determinism: 100%
Latency: Fast (1 pass)       Latency: Variable (1-3 passes)Latency: Zero extra passes
```

### 1. Constrained Decoding Mechanics (Outlines, SGLang, OpenAI `strict: true`)
Standard generation samples token $w_t \sim P(w_t \mid \dots)$.
In constrained decoding, a Context-Free Grammar (CFG) or JSON Schema compiles down to a deterministic finite automaton (DFA). At every generation step $t$, tokens that would violate the schema are masked out (logits set to $-\infty$) before softmax:

$$z'_{t, i} = \begin{cases} z_{t, i} & \text{if token } i \text{ is valid according to schema DFA} \\ -\infty & \text{otherwise} \end{cases}$$

This guarantees 100% syntactic compliance with zero parsing errors.

---

## 2. Engineering Function & Tool Calling Schemas

The prompt representation of tools dictates whether the LLM selects the correct tool and populates the arguments accurately.

### The 4 Golden Rules of Tool Schema Prompting:
1. **Descriptions Outweigh Parameter Types**: LLMs attend heavily to semantic docstrings. Explicitly describe *when to call* and *when NOT to call* the tool.
2. **Restrict String Arguments to Enums**: Never leave fields open-ended if the API expects a finite set of states (e.g. `status: "PENDING" | "ACTIVE" | "ARCHIVED"`).
3. **Embed Usage Preconditions**: Mention prerequisite tools directly in descriptions (e.g., `"Must call 'lookup_user_id' prior to this tool"`).
4. **Use Parallel Tool Calling Wisely**: When executing multi-item queries, enable parallel dispatch; when actions depend on prior state mutations, enforce sequential execution via prompt instructions.

### Hardened Tool Definition Example (JSON Schema):
```json
{
  "name": "execute_database_query",
  "description": "Executes read-only SQL queries against PostgreSQL warehouse. Do NOT use for INSERT, UPDATE, or DROP operations. Always verify column names using 'describe_tables' first.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The ANSI-compliant SELECT query. Must contain a LIMIT clause (maximum 1000)."
      },
      "timeout_seconds": {
        "type": "integer",
        "enum": [5, 10, 30],
        "description": "Query timeout in seconds. Default is 10."
      }
    },
    "required": ["query"],
    "additionalProperties": false
  },
  "strict": true
}
```

---

## 3. Self-Healing Schema Recovery Loop

Even with strict prompting, schema failures can happen on non-constrained endpoints. Implement an automated reflection repair prompt:

```
[LLM Call 1] ---> Malformed JSON ---> [Pydantic Validation Error]
                                             |
                                             v
[LLM Call 2: Repair Prompt] <----------------+
(Original Prompt + Broken JSON + Error Stack)
         |
         v
Valid Structured Output
```

### Production Repair Prompt Template:
```markdown
<system>
You are an automated JSON correction engine. Fix the provided output to adhere strictly to the target Pydantic schema.
Output ONLY the raw JSON object inside ```json ... ``` tags.
</system>

<target_schema>
{schema_definition}
</target_schema>

<failed_output>
{raw_model_response}
</failed_output>

<validation_errors>
{pydantic_error_trace}
</validation_errors>

<instruction>
Examine each validation error above. Correct the missing keys, invalid types, or misplaced delimiters. Retain all semantic content from the original output.
</instruction>
```
