# AI Agent Tool Use and Function Calling

## 1. OpenAI Function Calling Schema
Tools via 	ools array with 	ype: function. Each function specifies:
- 
ame: Identifier string.
- description: Functionality and triggering criteria.
- parameters: JSON Schema with properties, types, 
equired fields.
- strict: true enforces deterministic schema adherence via constrained decoding.

## 2. Anthropic Tool Use Format
Tools via 	ools parameter with 
ame, description, input_schema (JSON Schema).
- Model outputs 	ype: tool_use blocks (id, name, input).
- Client returns 	ype: tool_result with matching 	ool_use_id.

## 3. Parallel Tool Calling
Multiple distinct tool invocations in a single turn.
- OpenAI: parallel_tool_calls: true|false
- Minimizes latency in multi-step agent loops.

## 4. Tool Choice Forcing
- **auto**: Model chooses between text or tool call.
- **none**: Inhibits tool usage.
- **required/any**: Mandates calling a tool.
- **Forced Specific**: Constrains to one function.

## 5. Best Practices for Tool Descriptions
- Describe what tool does AND when NOT to call it.
- Avoid overlapping capabilities across tools.
- Use enum to bound discrete parameter options.
- Return structured, actionable error messages so agent can self-correct.