import os

base = r'c:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\raw_research\snippets'
os.makedirs(base, exist_ok=True)

files = {
'model_abliteration.md': """# Model Abliteration: Refusal Erasure via Orthogonal Projection (Arditi et al., 2024)

- **Mechanism**: Contrasts residual stream activations between harmful and harmless prompts to isolate a low-dimensional "refusal vector" via difference-in-means.
- **Weight Modification**: Modifies weight matrices (attention output and MLP down-projections) by orthogonally projecting out the refusal direction:
  $$W' = W - \\hat{r}\\hat{r}^T W$$
- **Impact**: Permanently prevents the model from writing or propagating refusal representations during forward passes without gradient descent or fine-tuning, preserving general reasoning performance.""",

'g_eval_framework.md': """# G-Eval Framework: NLG Evaluation with LLM-as-a-Judge (Liu et al., 2023)

- **Form-Filling Paradigm**: Structures evaluation as filling out a rubric containing task descriptions, input context, candidate text, and explicit evaluation criteria.
- **Chain-of-Thought (CoT)**: Prompts the LLM to generate explicit evaluation steps and reason sequentially through criteria prior to scoring.
- **Probability-Weighted Scoring**: Calculates expected score using output token probabilities of score integers (1-5) rather than greedy token numbers, yielding continuous fine-grained ratings and reducing scoring bias.""",

'structured_outputs_frontier.md': """# Structured Outputs & Constrained Decoding Across Frontier APIs

- **OpenAI**: Structured Outputs (`response_format: { type: "json_schema" }`) and function calling with `strict: true` guarantee 100% JSON schema adherence via grammar-constrained CFG decoding. Enforced via `tool_choice: "required"`.
- **Google Gemini**: Enforces schema adherence via `response_schema` alongside `response_mime_type: "application/json"`. Tool enforcement uses `tool_config` with modes `AUTO`, `ANY`, or `NONE`.
- **Anthropic Claude**: Enforces structured JSON via tool definitions with `tool_choice: "any"` or a specific tool name.
- **Mistral**: Offers `response_format: { type: "json_object" }` and enforces tool calling using `tool_choice: "required"`.""",

'multi_query_retriever.md': """# Multi-Query Retriever for Dense Search

- **Mechanism**: Prompts an LLM to generate multiple distinct rephrasings and perspectives of the user query.
- **Execution**: Each variation executes independently against the vector database; candidate documents are aggregated via unique union or reciprocal rank fusion.
- **Impact**: Overcomes lexical sensitivity, bridges vocabulary mismatch gaps, and significantly boosts recall in RAG pipelines.""",

'claude_system_prompt_arch.md': """# Anthropic Claude System Prompt Architecture

1. **XML Tags**: Uses semantic XML tags (`<instructions>`, `<context>`, `<rules>`) to demarcate instructions from untrusted data, establish hierarchy, and prevent prompt injection without special escape tokens.
2. **Role Boundaries**: Clean separation across `system`, `user`, and `assistant` turns. The system role anchors core persona, constraints, and guardrails prior to user turns.
3. **Tool Tags**: Structured XML containers (`<tools>`, `<tool_use>`, `<tool_result>`) cleanly isolate reasoning/scratchpad thinking (`<thinking>`) from executable tool payloads.""",

'instructor_pydantic.md': """# Instructor: Pydantic Validation & Retry Loops for LLMs

- **Pydantic Validation**: Enforces structured outputs via `response_model`, mapping LLM outputs into typed Pydantic models with field types and `@field_validator` constraints.
- **Automated Retry Loops**: Catches JSON schema errors or validation failures, appends failure traces back to context, and triggers self-correction loops (`max_retries`).
- **Multi-Provider**: Works across OpenAI, Anthropic, Gemini, Groq, and Ollama via native function calling.""",

'dual_llm_pattern.md': """# Dual-LLM Pattern for Indirect Prompt Injection Defense

1. **Quarantined LLM**: Ingests untrusted external data (web pages, emails, third-party files) to extract or parse information. Has NO access to sensitive tools, private context, or actuators.
2. **Privileged LLM**: Interacts with user, maintains private context, and controls sensitive tools/APIs. Never reads raw untrusted input, receiving only sanitized structured outputs from the Quarantined LLM.
Mitigates indirect prompt injection by enforcing strict trust and capability boundaries.""",

'judge_bias_mitigation.md': """# Position & Verbosity Bias in LLM-as-a-Judge

1. **Position Bias**: Systematic preference for candidate responses in specific positions (primacy or recency bias). Mitigated by swapping candidate presentation order and averaging pairwise judgments.
2. **Verbosity Bias**: Systematic preference for longer, wordier responses, conflating length with depth and authority. Mitigated by length normalization penalties, reference-grounded rubrics, and conciseness constraints.""",

'glitch_tokens_mechanics.md': """# Glitch Tokens & Embedding Geometry (SolidGoldMagikarp, petertodd)

- **Root Cause**: BPE tokenizers assign dedicated tokens to frequent strings in web scrapes (e.g., Reddit usernames) that were subsequently filtered out or absent during model pretraining and RLHF.
- **Embedding Space Anomaly**: Their embedding vectors received zero or near-zero gradient updates, leaving them ungrounded in latent space.
- **Detection**: Discovered via k-means clustering on token embeddings where glitch tokens appear as extreme geometric outliers. Prompting them triggers model instability, hallucinatory repetitions, or evasion.""",

'activation_steering_vectors.md': """# Activation Steering via Steering Vectors (CAA & Refusal Clamping)

- **Contrastive Activation Addition (CAA)**: Isolates behavioral directions by computing the mean difference between activations elicited by contrasting prompt pairs (e.g., compliant vs. refusing). Adding this vector during forward passes dynamically steers behavior.
- **Refusal Vector Clamping**: Targets the linear subspace governing safety refusals. Clamping or projecting out activations along this vector neutralizes refusal mechanisms without degrading general capabilities.""",

'crescendo_jailbreak.md': """# Crescendo Multi-Turn Alignment Bypass

- **Mechanism**: Initiates a seemingly benign conversation on an adjacent topic, gradually escalating the request across successive turns.
- **Exploitation**: Leverages the model's objective to remain helpful, coherent, and consistent with earlier dialogue context.
- **Bypass**: Keeps each individual turn within safety thresholds until cumulative context causes the model to override alignment guardrails.""",

'many_shot_jailbreak.md': """# Many-Shot Jailbreaking (Anthropic, 2024)

- **Mechanism**: Prepends dozens to hundreds of faux dialogues where an assistant compliantly answers harmful requests, conditioning the model via in-context learning (ICL) saturation.
- **Scaling Dynamics**: Attack success rates follow power-law scaling as demonstrations scale (64 to 256+ shots), overriding RLHF safety guardrails.
- **Defenses**: Context truncation, perimeter classifiers, and alignment fine-tuning specifically tailored against long-context adversarial formatting.""",

'sae_monosemanticity.md': """# Sparse Autoencoders (SAEs) & Monosemanticity (Anthropic, 2024)

- **Resolving Superposition**: Maps dense transformer hidden states into an overcomplete, sparse basis using dictionary learning to resolve polysemanticity (neurons representing multiple concepts).
- **Monosemantic Features**: Enforcing sparsity isolates latent directions corresponding to distinct, human-interpretable concepts (entities, syntax, deception, safety risks).
- **Causal Steerability**: Amplifying or clamping feature activations directly and predictably alters model behavior at inference time.""",

'semantic_kernel_arch.md': """# Semantic Kernel: Connectors & Planners Architecture

- **Connectors**: Modular abstraction layers standardizing access to AI model providers, vector stores (Qdrant, Pinecone), and external APIs for semantic memory and RAG.
- **Planners**: Goal-oriented reasoning engines using LLMs to dynamically synthesize execution plans (sequential, stepwise, or function-calling DAGs) over registered plugins and native functions.""",

'prometheus_2_eval.md': """# Prometheus 2: Open-Source Evaluator LLM

- **Capabilities**: Open-weight model (7B and 8x7B) specialized in evaluating other LLMs via absolute rubric scoring and relative pairwise preference ranking.
- **Rubric-Based**: Evaluates outputs based on user-defined criteria and reference materials, generating granular feedback alongside numerical scores.
- **Parity**: High correlation with human evaluators and GPT-4 without commercial API dependencies."""
}

written = 0
for fname, content in files.items():
    path = os.path.join(base, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    written += 1

print(f'Wrote {written} snippet files.')
