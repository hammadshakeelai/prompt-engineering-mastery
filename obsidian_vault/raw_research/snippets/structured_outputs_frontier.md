# Structured Outputs & Constrained Decoding Across Frontier APIs

- **OpenAI**: Structured Outputs (`response_format: { type: "json_schema" }`) and function calling with `strict: true` guarantee 100% JSON schema adherence via grammar-constrained CFG decoding. Enforced via `tool_choice: "required"`.
- **Google Gemini**: Enforces schema adherence via `response_schema` alongside `response_mime_type: "application/json"`. Tool enforcement uses `tool_config` with modes `AUTO`, `ANY`, or `NONE`.
- **Anthropic Claude**: Enforces structured JSON via tool definitions with `tool_choice: "any"` or a specific tool name.
- **Mistral**: Offers `response_format: { type: "json_object" }` and enforces tool calling using `tool_choice: "required"`.