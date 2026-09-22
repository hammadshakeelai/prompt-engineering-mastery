# DeepSeek-R1 Prompting Guidelines

- **Zero-Shot Directness**: State problems directly; do not force artificial reasoning prompts or persona gymnastics.
- **Outcome Specification**: Define target output schemas (XML tags, JSON format) clearly.
- **Temperature Setting**: Recommended temperature 0.6 for creative/extended reasoning; lower for deterministic code/math tasks.
- **Structured Context**: Wrap reference context in delimiters (`<context>`, `<problem>`) to prevent prompt drift.