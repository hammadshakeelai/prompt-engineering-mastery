# Thread of Thought (ThoT) (Zhou et al., 2023)

- **Incremental Segment Summary**: Prompts model to divide context into manageable chunks, analyzing and summarizing sequentially before final synthesis.
- **Chaotic Context Handling**: Mitigates "lost-in-the-middle" and distractor corruption in RAG/multi-turn dialogue by maintaining balanced attention across all document sections.
- **Vs. CoT**: CoT focuses on logical deduction; ThoT acts as a cognitive digest filter prioritizing evidence extraction in noisy contexts.