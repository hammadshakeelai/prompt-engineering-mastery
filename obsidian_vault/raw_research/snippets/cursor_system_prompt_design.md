# Cursor IDE System Prompt Design for Code Agents

- **Two-Tier Diff Architecture**: Avoids raw code block output; uses surgical file-editing tools with `// ... existing code ...` placeholders, offloading patch application to an ultra-fast secondary model.
- **Bounded Autonomy**: 3-iteration limit on fixing linter errors before prompting user; enforces user approval for terminal execution and file citation boundaries (`startLine:endLine:filepath`).