# OpenAI o1 & o3 Prompting Rules

1. **No Manual CoT**: Avoid *"think step-by-step"*; models reason autonomously via internal hidden CoT. Manual CoT prompts degrade performance.
2. **Clean Constraints**: Plainly define goals, formatting rules, edge conditions, and acceptance criteria.
3. **Structured Delimiters**: Use Markdown or XML tags to cleanly separate inputs and schemas.
4. **Format-Focused Examples**: Provide clean input-output pairs without intermediate rationale traces.