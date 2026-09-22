# Lost in the Middle (Liu et al., TACL 2024)

**Citation:** arXiv:2307.03172

## Core Finding: U-Shaped Performance Curve
LLMs access information best at context boundaries:
- **Primacy bias:** Information at the start of context.
- **Recency bias:** Information at the end of context.
- **Middle degradation:** Performance near closed-book baselines for middle-positioned evidence.

## Scope
Affects GPT-3.5/4, Claude, MPT-30B, LLaMA-2 — even models marketed for long context.

## Mechanics
Causal masking and positional encodings bias attention toward initial (structural anchors) and terminal (generation prefix) tokens.

## Implications for RAG
1. **Perimeter Reordering:** Top chunks at start/end; lowest-ranked in the center trough.
2. **Prompt Sandwiching:** System prompt first, docs in body, restate query + schema at end.
3. **Aggressive Pruning:** Restrict to top-5 chunks or use LLMLingua compression.