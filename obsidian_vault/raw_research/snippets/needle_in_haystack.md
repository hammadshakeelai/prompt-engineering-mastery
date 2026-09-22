# Needle in a Haystack (NIAH) Benchmark

Introduced by Greg Kamradt (late 2023). Tests LLM long-context retrieval by hiding a specific fact (needle) in background text (haystack, typically Paul Graham essays).

## Methodology
Two-axis 2D retrieval heatmap:
1. Context Length: 1k to 128k+ tokens.
2. Document Depth: Needle position from 0% (top) to 100% (bottom) of context.

## Key Model Results
- GPT-4 Turbo: Reliable to ~64k tokens, degradation toward 128k especially at mid-depths.
- Claude 2.1 (200k): Pronounced lost-in-the-middle effect at 50k-150k mid-depths. Claude 3 (Opus/Sonnet) resolved this, achieving >99% recall across 200k.
- Gemini 1.5 Pro: Near-flawless (>99%) recall across 1M-2M token contexts across all depths and modalities.

## Implications for Prompt Design
1. Place critical instructions at beginning OR end, not buried in the middle.
2. Prompt Scaffolding: Ask model to quote exact context to improve attention anchoring.
3. Selective Context: NIAH tests simple lookup, not multi-hop reasoning. Targeted RAG > blind context dumping.