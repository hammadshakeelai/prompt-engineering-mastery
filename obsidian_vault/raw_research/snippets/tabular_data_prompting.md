# Tabular and Structured Data Prompting

## Table Representation Formats
- Markdown Tables: Strongest inductive bias and highest reasoning accuracy due to rich web pretraining. Table syntax tokens (|, -) increase token overhead.
- CSV: Highest token efficiency (2-3x fewer tokens than Markdown/JSON), ideal for long contexts. Lower relational accuracy on wide tables.
- JSON/KV Pairs: Explicit key-attribute alignment for nested/sparse data, but heaviest token redundancy.

## Specialized Frameworks
- Chain-of-Table (Wang et al., ICLR 2024): Decomposes tabular reasoning into step-by-step table evolution. LLM iteratively plans and executes operations (add_column, select_row, group_by), passing transformed sub-tables as the reasoning chain. Outperforms standard CoT on WikiTQ and TabFact.
- TableGPT/TableGPT2: Domain-specialized LLMs fine-tuned on tabular corpora. Integrates code interpreters for unified table manipulation, QA, and visualization.
- Code-Interpreted Agents (PandasAI): Routes reasoning through code execution - LLM generates and executes deterministic Python/pandas operations rather than in-context string ingestion.

## Text-to-SQL Prompting Patterns
- Schema Linking and Pruning: Inject minimal DDL statements, key constraints, and 3-5 representative row samples. Token-to-column linking resolves ambiguous NL references.
- Multi-Step Decomposition (DIN-SQL): Schema linking -> query classification (simple vs. nested) -> dialect SQL drafting.
- Few-Shot Selection and Self-Correction (DAIL-SQL): Dynamic exemplar selection based on question-schema similarity + execution error feedback for iterative self-repair.