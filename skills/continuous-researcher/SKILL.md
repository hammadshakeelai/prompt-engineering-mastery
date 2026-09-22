---
name: continuous-researcher
description: Autonomous protocol for conducting deep, empirical, zero-hallucination scientific research into AI models, mechanistic interpretability, and prompting paradigms.
---

# Continuous Researcher Skill

Use this skill when autonomously surveying academic literature, gathering empirical benchmark data, detecting conceptual gaps in the knowledge base, and synthesizing atomic research notes.

## 1. Research Protocol

1. **Query Formulation**:
   - Always search for specific paper titles, author names, conference proceedings (NeurIPS, ICLR, ICML, ACL), and exact technical terms.
   - Avoid generic web queries; use targeted phrases with publication years (e.g., `"in-context learning" "PAC learning" 2024 OR 2025`).

2. **Fact Verification & Anti-Hallucination**:
   - Extract exact authors, conference/journal citations, arXiv identifiers, and empirical performance metrics.
   - Cross-check counter-arguments and metric controversies (e.g., emergent abilities debate, CoT faithfulness limitations).

3. **Atomic Snippet Synthesis**:
   - Save new findings in `obsidian_vault/raw_research/snippets/<topic_name>.md`.
   - Maintain high density: 200–350 words, structured into clear subheadings, equations, and benchmark numbers.
   - Link back to related notes using Obsidian wiki-links (`[[topic_name]]`).

4. **Vault & Remote Synchronization**:
   - Update `obsidian_vault/000_Master_Brain_Index.md` using `scripts/build_obsidian_index.py`.
   - Commit and push to Git regularly.
