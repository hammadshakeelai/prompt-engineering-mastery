# Anthropic Claude System Prompt Architecture

1. **XML Tags**: Uses semantic XML tags (`<instructions>`, `<context>`, `<rules>`) to demarcate instructions from untrusted data, establish hierarchy, and prevent prompt injection without special escape tokens.
2. **Role Boundaries**: Clean separation across `system`, `user`, and `assistant` turns. The system role anchors core persona, constraints, and guardrails prior to user turns.
3. **Tool Tags**: Structured XML containers (`<tools>`, `<tool_use>`, `<tool_result>`) cleanly isolate reasoning/scratchpad thinking (`<thinking>`) from executable tool payloads.