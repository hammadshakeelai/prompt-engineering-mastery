import os

base = r'c:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\raw_research\snippets'
os.makedirs(base, exist_ok=True)

files = {
'synthetic_data_generation.md': """# Synthetic Data Generation for LLM Training

## 1. Self-Instruct (Wang et al., 2022)
Bootstraps instruction-following datasets via iterative semi-automated pipeline.
- Seed Tasks: 175 human-written task exemplars.
- Generation and Filtering: LLM (GPT-3) generates instruction-input-output triplets. Pruned via ROUGE-L similarity (<0.7) and heuristic filters.
- Impact: Produced 52K instructions, blueprint for cost-effective synthetic alignment.

## 2. Stanford Alpaca (Taori et al., 2023)
- Used text-davinci-003 to generate 52,000 instruction-response pairs from 175 seed tasks for under $500.
- Fine-tuned LLaMA-7B on synthetic dataset, proving lightweight open models can approximate proprietary instruction-following behaviors.

## 3. WizardLM: Evol-Instruct (Xu et al., 2023/2024)
Scales prompt complexity via iterative evolution:
- In-Depth Evolution: Adds constraints, deepens reasoning, concretizes abstract concepts, chains complex logic.
- In-Breadth Evolution (Mutation): Broadens topic coverage, creates novel domain tasks.
- Enables models to master sophisticated coding and mathematical reasoning.

## 4. Orca 2: Synthetic CoT and Strategy Selection (Mitra et al., 2023)
Teaches Small Language Models diverse reasoning strategies (step-by-step, recall-then-generate, direct answering).
- Prompt Erasure: Teacher (GPT-4) gets rich system prompts guiding detailed CoT traces. During student training, guidance prompt is stripped. Student must autonomously synthesize internal reasoning paths.""",

'prompt_formatting.md': """# Prompt Format and Delimiter Effects on LLM Performance

## 1. Delimiter Syntax Across Architectures
- XML Tags: Explicit opening/closing boundaries, hierarchical nesting, metadata attributes. Claude (Anthropic) is explicitly fine-tuned on XML - superior instruction-data disambiguation.
- Markdown Headers (#, ##): Delineate macro-level section hierarchy. GPT-4 family has strong affinity for Markdown from RLHF and developer tuning.
- Triple Backticks: Industry standard for verbatim code/JSON/reference docs. Lacks semantic labeling and hierarchical nesting vs. XML.
- HTML Formatting: Less reliable - web-crawl noise and auto-escape mechanisms often mangle HTML tokens.

## 2. Empirical Evidence

### FORMATSPREAD (Sclar et al., ICLR 2024)
Semantically equivalent prompt format perturbations on LLaMA-2 caused accuracy swings of up to 76 percentage points solely from superficial delimiter and formatting variations.

### He et al. (2024)
GPT-3.5 varied by up to 42% across Markdown/JSON/YAML/plain text formats.
GPT-4 favored Markdown for MMLU reasoning (81.2% Markdown vs. 73.9% JSON).

### Delimiter Token Choice
Modifying single character separating few-shot examples altered MMLU accuracy by 18.3-29.4% across Llama-3.1 and Qwen2.5.

### Prompt Injection Defense (McMillin, 2026)
XML delimiters reduce injection success from 18.2% to 9.9% (1.0% combined with sandwich framing).""",

'negative_prompting.md': """# Negative Prompting and Contrastive Decoding for LLMs

## 1. Negative Prompting / Steering
Steers models away from hallucinations, repetition, verbosity, toxic personas.
In inference steering (CFG for LLMs), decoding contrasts positive vs. negative prompt logits:
logit(y_t) = logit_pos(y_t) + gamma * (logit_pos(y_t) - logit_neg(y_t))
Penalizes probability mass associated with unwanted concepts.

## 2. Contrastive Decoding (Li et al., 2022)
Contrasts output distributions of Expert (large) vs. Amateur (small) model:
CD Score: log p_expert(y_t | y<t) - alpha * log p_amateur(y_t | y<t)
Amateur models disproportionately favor generic/vacuous tokens; logit subtraction suppresses these.
Adaptive plausibility constraint filters tokens below threshold tau * max_w p_expert(w).

## 3. DoLa: Decoding by Contrasting Layers (Chuang et al., 2023)
Eliminates separate amateur model by contrasting internal transformer layers.
- Premise: Lower layers encode surface/linguistic statistics; deeper layers encode factual knowledge.
- Mechanism: Dynamically identifies premature layer with high JSD divergence from final layer.
- Decoding: Subtracts premature layer logits from final layer logits to amplify factual signals and suppress hallucinations.""",

'raft_ra_dit.md': """# RAFT and RA-DIT: Retrieval-Augmented Fine-Tuning

## 1. RAFT (Zhang et al. / Gorman et al., 2024)
Trains LLMs for domain-specific "open-book" QA - extracting evidence from context while ignoring noise.
- Training Recipe: Each sample pairs question with oracle documents (containing answer) + distractor documents (that don't). In subset, oracle documents withheld.
- Model generates CoT answers citing verbatim excerpts from relevant docs.
- Builds robustness against retriever failures and hallucinations.

## 2. RA-DIT (Lin et al., 2023 - Meta AI)
Dual optimization of generator (LLM) and dense retriever without full end-to-end backpropagation.
- LM Tuning: Instruction-tunes LLM conditioned on retrieved passages.
- Retriever Tuning: Uses LM output distribution (KL-divergence feedback) to train dual-encoder retriever.

## 3. RAG-Specific vs. Standard Fine-Tuning
- Standard SFT: Forces factual knowledge into model weights (parametric memorization).
- RAG Fine-Tuning: Trains non-parametric extraction - how to search, ground, and cite from dynamic inputs.
- Distractor Tolerance: RAG tuning explicitly conditions on noisy/conflicting documents.
- Retriever Alignment: Co-adapts retriever distributions with LM context-conditioning behavior.""",
}

written = 0
for fname, content in files.items():
    path = os.path.join(base, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    written += 1

print(f'Wrote {written} files.')
