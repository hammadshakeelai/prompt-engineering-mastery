---
name: vault-synthesizer
description: Maintaining, cross-linking, indexing, and organizing an atomic Obsidian knowledge vault of technical AI and prompt engineering concepts.
---

# Vault Synthesizer Skill

Use this skill to maintain graph integrity, generate structural indexes, prevent fragmentation, and cross-reference related concepts across the knowledge base.

## 1. Vault Organization Principles

1. **Atomic Modularity**:
   - Each snippet file in `obsidian_vault/raw_research/snippets/` focuses on one unified algorithmic or theoretical concept.
   - Files are named in snake_case (e.g., `flashattention_2_vs_3.md`, `mamba_ssm.md`).

2. **Bidirectional Cross-Linking**:
   - When discussing related mechanisms, reference peer notes using standard markdown wiki-links: `[[topic_name]]`.
   - Ensure foundational concepts (e.g., `[[in_context_learning]]`, `[[kv_cache_mechanics]]`) serve as anchor hubs for specialized techniques.

3. **Graph Index Regeneration**:
   - Run `python scripts/build_obsidian_index.py` whenever files are added or modified.
   - Verify that `obsidian_vault/000_Master_Brain_Index.md` contains updated categories, tags, and complete link coverage.
