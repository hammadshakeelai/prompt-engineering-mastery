# MemGPT / Letta Hierarchical Agent Memory

Models LLM memory after an operating system virtual memory hierarchy:
1. **In-Context (RAM)**:
   - Core Memory: Fixed-size blocks (persona, user profile) pinned in system prompt, editable via function calls.
   - Working Context: FIFO conversational message buffer.
2. **Out-of-Context (Disk)**:
   - Recall Memory: Searchable conversational history log.
   - Archival Memory: Vector database for arbitrary knowledge retrieval.
Agent autonomously pages, edits, and searches across tiers via tool calls for unbounded persistent state.