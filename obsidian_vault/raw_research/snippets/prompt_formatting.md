# Prompt Format and Delimiter Effects on LLM Performance

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
XML delimiters reduce injection success from 18.2% to 9.9% (1.0% combined with sandwich framing).