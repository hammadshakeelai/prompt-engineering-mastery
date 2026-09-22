# Instructor: Pydantic Validation & Retry Loops for LLMs

- **Pydantic Validation**: Enforces structured outputs via `response_model`, mapping LLM outputs into typed Pydantic models with field types and `@field_validator` constraints.
- **Automated Retry Loops**: Catches JSON schema errors or validation failures, appends failure traces back to context, and triggers self-correction loops (`max_retries`).
- **Multi-Provider**: Works across OpenAI, Anthropic, Gemini, Groq, and Ollama via native function calling.