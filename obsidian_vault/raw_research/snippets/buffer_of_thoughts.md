# Buffer of Thoughts (BoT) - Yang et al. (NeurIPS 2024)

Citation: arXiv:2406.04271

## Core Motivation
ToT achieves high accuracy but requires 100+ queries.
Standard CoT generates thoughts from scratch without cross-task memory.
BoT introduces thought-augmented reasoning with a persistent meta-buffer.

## Architecture
1. Problem Distiller: Extracts critical task specifications, constraints, and mathematical relationships.
2. Meta-Buffer (Thought-Template Library): Stores high-level thought-templates - distilled meta-thoughts and algorithmic patterns (dynamic programming, divide-and-conquer, proof-by-contradiction) abstracted from previous solutions.
3. Buffer-Manager: Maintains, refines, and deduplicates templates. Prevents memory drift.
4. Adaptive Instantiation: Retrieves most relevant template and instantiates into concrete multi-step reasoning path.

## Efficiency and Performance
- Consumes ~12% of the compute cost of Tree of Thoughts.
- Outperforms CoT and matches/surpasses ToT on Game of 24, 3D spatial reasoning, Checkmate-in-One.
- Enables smaller models (Llama-3-8B) to outperform larger models (Llama-3-70B) via structured template guidance.

## Vs. Step-Back Prompting
Step-Back abstracts on-the-fly. BoT catalogs persistent distilled abstractions for cross-task retrieval.