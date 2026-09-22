# XGrammar: Fast Structured Generation & Bitmask Automata (2024–2026)

**Citation:** *XGrammar: Flexible and Efficient Structured Generation to Support Diverse Engines and Hardware* (CMU / MLC-AI, arXiv:2411.15100)

## The Latency Bottleneck of Constrained Decoding
Traditional constrained generation (Outlines regex DFAs, Lark CFG parsers) checks candidate tokens across full vocabularies (32k–128k+) at every generation step, introducing 10x–50x inference latency penalties and preventing deployment in high-throughput engines.

## 1. Bitmask Transition Automata
XGrammar compiles context-free grammars (CFGs) and JSON schemas into compact bitmask transition matrices:
- **Offline Bitmask Compilation:** Translates grammar states into dense binary masks directly matching token IDs. Logit masking reduces to a parallel bitwise AND operation on GPU registers rather than expensive string matching or AST traversal.
- **Vocabulary Partitioning:** Separates context-independent tokens (predictable offline) from context-dependent tokens, reducing runtime evaluation by up to 99%.

## 2. Integration with Production Serving Engines
Adopted as the primary structured generation backend in **vLLM, SGLang, TensorRT-LLM, and MLC-LLM**:
- **Persistent Stacks:** Synchronizes grammar state stacks with the engine's KV-cache, preserving state across multi-turn sessions and speculative decoding drafts.
- **Near-Zero Latency Overhead:** Achieves up to 100x speedup over earlier grammar libraries, running structured JSON generation at virtually the same throughput as unconstrained generation.
- **Jump-Forward Acceleration:** In conjunction with SGLang, deterministic grammar transitions enable speculative batch emissions of structural syntax without forward-pass sampling.
