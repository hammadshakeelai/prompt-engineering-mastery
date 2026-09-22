# AutoGen Conversational Topologies

1. **Group Chat**: Multiple agents share a single conversation managed by a `GroupChatManager`. Next-speaker selection can be round-robin, random, manual, or dynamically selected by an LLM based on task context.
2. **Sequential Chat**: Agents interact in a predefined linear pipeline. Output/summary of one conversation feeds as input into the next.
3. **Nested Chat**: An agent triggers an internal, subordinate multi-agent dialogue to complete a specific sub-task or review before replying to the main chat, returning only a synthesized result.