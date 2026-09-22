# Analogical Prompting (Yasunaga et al., ICLR 2024)

Citation: arXiv:2310.01714 (Stanford / Google DeepMind)

## Connection to Human Analogical Reasoning
Humans resolve novel problems by retrieving structurally analogous past experiences and transferring reasoning strategies.
Analogical prompting mimics this by guiding LLMs to recall relevant problem schemas from parametric memory.

## Self-Generated Analogous Problems
Traditional few-shot CoT relies on static human-labeled exemplars.
Analogical prompting directs the LLM to autonomously recall/generate 3-5 relevant exemplars tailored on-the-fly:
- Analogous problems
- Step-by-step reasoning rationales
- Solutions matching the input problem's sub-domain, difficulty, and algorithmic patterns

## Outperforming Few-Shot CoT
- Consistently outperforms 0-shot CoT and matches/surpasses manual few-shot CoT.
- Benchmarks: GSM8K, MATH, Codeforces competitive programming.
- Dynamic Context Adaptation: Self-generated exemplars cover long-tail theorems and data structures.
- Zero Labeling Overhead: Eliminates need for curated few-shot exemplar repositories.