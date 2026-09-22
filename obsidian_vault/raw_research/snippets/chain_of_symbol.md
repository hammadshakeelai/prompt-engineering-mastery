# Chain-of-Symbol (CoS) Prompting - Hu et al. (2023)

Citation: arXiv:2305.10276

## Core Motivation
LLMs struggle with spatial planning in natural language due to semantic ambiguity and compounding hallucinations.
Standard CoT uses verbose natural language scratchpads, introducing spatial confusion during intermediate state tracking.

## Methodology
Replaces natural language scratchpads with condensed symbolic representations:
- Symbolic State Representation: Entities, locations, and spatial relations encoded as concise symbols/tuples (coordinates, connection symbols, state tokens) rather than prose narratives.
- Intermediate reasoning steps update state transitions symbolically, providing an explicit mental map.
- Plug-and-play: Pure prompt engineering, no fine-tuning required.

## Empirical Gains
- Brick World (Spatial Manipulation): ChatGPT 31.8% (CoT) -> 92.6% (CoS), +60.8% absolute gain.
- Natural Language Navigation: Outperforms CoT by preserving spatial coherence and path validity.
- Spatial QA (SPARTUN): Consistently outperforms CoT, reducing relational errors.
- Efficiency: Reduces intermediate reasoning tokens by up to 65.8%, lowering inference latency.