# Universal Self-Consistency (Chen et al., 2023)

Citation: arXiv:2311.17311

## Core Concept
Standard Self-Consistency uses exact-match majority voting, restricting to closed-domain discrete answers.
Universal Self-Consistency (USC) extends to arbitrary free-form text by replacing parsing with model-based semantic consensus.

## Mechanism: LLM as Selector
1. Diverse Sampling: Generate N candidate responses using temperature sampling + CoT.
2. Contextual Aggregation: Concatenate original query + all candidate responses into a single prompt.
3. Consensus Selection: LLM selector evaluates semantic consistency, mutual agreement, reasoning validity across the pool.

Evaluating mutual consistency is cognitively easier for LLMs than verifying absolute correctness in isolation.

## Advantages
- Free-Form Text: Works on summarization, open-ended QA, creative reasoning without discrete answer extraction.
- No Parsers: Eliminates fragile regex, keyword extractors, and external execution verifiers.
- Breadth: Matches standard majority voting on math, competitive with execution-based voting on code.