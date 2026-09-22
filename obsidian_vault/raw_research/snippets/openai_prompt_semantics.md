# GPT-4o System Prompt Structure & Developer Message Semantics

**Modular Architecture**:
1. Identity & Role: Core persona and capabilities.
2. Behavioral Constraints: Tone, style, Markdown formatting, conciseness.
3. Safety & Policy: Guardrails and refusal boundaries.
4. Tool Definition: Function-calling interfaces and environment metadata.

**Developer Role Semantics**:
- Commands from the `developer` role take strict precedence over `user` messages.
- Separates immutable application directives and governance from untrusted user inputs, mitigating prompt injection.