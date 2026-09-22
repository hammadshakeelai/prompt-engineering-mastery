---
name: prompt-architect
description: Comprehensive framework for analyzing, designing, optimizing, and evaluating prompt architectures across reasoning, agentic workflows, and frontier models (o1/o3, Claude 3.5, Gemini 1.5/2.0, DeepSeek-R1).
---

# Prompt Architect Skill

Use this skill when designing, debugging, or optimizing prompts for single-turn reasoning, multi-turn dialogues, autonomous agent scaffolds, or structured output generation.

## 1. Architectural Taxonomy Selection

Before drafting a prompt, select the optimal cognitive topology based on task demands:

| Task Class | Recommended Architecture | Delimiters & Schemas | Key Pitfall to Avoid |
| :--- | :--- | :--- | :--- |
| **Complex Math / Logic** | Program-Aided Language (PAL) or Self-Consistency CoT ($k=10, T=0.7$) | Code blocks or `<thinking>` wrappers | Superfluous verbal reasoning on simple arithmetic |
| **Frontier Reasoning Models (o1/o3/R1)** | Direct Objective + Constraint Specifications | Plain Markdown or XML tags | Appending *"think step-by-step"* (degrades native reasoning) |
| **Noisy / Long-Context RAG** | Contextual Retrieval + Thread-of-Thought (ThoT) | Chunk metadata headers + `<context>` tags | Lost-in-the-middle degradation (put queries at prompt boundary) |
| **Complex Multi-Step Goals** | Plan-and-Solve (PS+) or Hierarchical Sub-task Routing | Ordered sub-tasks + typed JSON handoff | Compounding error across monolithic single-pass generations |
| **Strict Data Extraction** | Constrained Grammar / Pydantic Schema via Instructor/Outlines | Typed JSON Schema | Loose natural language format requests without schema enforcement |

## 2. Structural Prompt Scaffolding Protocol

1. **System Persona & Boundary**: Define operational identity without generic conversational fluff. Specify refusal conditions and edge-case behaviors.
2. **Context Delimitation**: Enclose all external context, retrieved passages, or user documents inside explicit XML tags (`<context>`, `<document id="...">`).
3. **Exemplar Selection (Few-Shot)**:
   - Ensure balanced class distributions to prevent prior label bias.
   - Use maximal diversity (MMR) across exemplars.
   - For reasoning tasks, ensure intermediate steps reflect true causal logic.
4. **Output Constraint & Schema**:
   - Provide an exact schema or target format specification at the prompt terminator.
   - State formatting requirements in both positive (what to include) and negative (what to omit) terms.
