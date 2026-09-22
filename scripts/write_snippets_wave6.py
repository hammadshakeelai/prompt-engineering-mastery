import os

base = r'c:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\raw_research\snippets'
os.makedirs(base, exist_ok=True)

files = {
'self_rag_ext.md': """# Self-RAG: Self-Reflective Retrieval-Augmented Generation

Enhances factuality by training an LLM to output special reflection tokens:
- `[Retrieve]`: Dynamically decides whether external knowledge retrieval is necessary (`yes`, `no`, `continue`).
- `[IsRel]`: Evaluates whether a retrieved passage contains relevant context to address the prompt.
- `[Critique]` (`[IsSup]`, `[IsUse]`): Assesses generated responses. `[IsSup]` verifies whether claims are supported by evidence; `[IsUse]` rates overall usefulness.

Reflection token probabilities guide beam search during inference, filtering irrelevant docs and selecting the most grounded text.""",

'langgraph_patterns.md': """# LangGraph State Machine Patterns

LangGraph models workflows as cyclical state machines where nodes read and update shared, typed state:
- **Reducers**: Specify how state updates merge. Custom reducer functions (e.g., `operator.add` via `Annotated`) append or modify specific state keys.
- **Checkpointers**: Provide persistence by saving state snapshots at each superstep. Enables session continuity, fault tolerance, and time travel (rewinding to prior states).
- **Human-in-the-Loop**: Leverages checkpointers to pause execution via breakpoints or dynamic `interrupt()` calls for operator review, editing, or approval before resuming.""",

'contextual_retrieval.md': """# Contextual Retrieval (Anthropic 2024)

Solves context loss caused by splitting documents into isolated chunks.
Standard chunking strips surrounding document context, causing retrieval failures on pronouns or undefined metrics.
- **Mechanism**: Prompts an LLM (Claude 3 Haiku) with the full document to generate a concise, chunk-specific explanatory context (50-100 tokens), prepended directly to each chunk prior to embedding and indexing.
- **Impact**: When combined with hybrid search (dense + BM25) and reranking, reduces retrieval failure rates by up to 67%. Cost-effective via prompt caching.""",

'rag_fusion_ext.md': """# RAG-Fusion & Multi-Query Expansion

Overcomes single-query retrieval limitations through multi-query expansion and rank-based aggregation:
1. **Query Expansion**: LLM generates multiple diverse rephrasings and perspectives of the original prompt.
2. **Parallel Retrieval**: Each variation independently retrieves candidate documents.
3. **Reciprocal Rank Fusion (RRF)**: Merges and re-ranks results:
   $$\\text{RRF Score}(d) = \\sum_{q} \\frac{1}{k + \\text{rank}(d, q)}$$
   Prioritizes documents appearing consistently near the top across multiple queries.
4. **Generation**: Top-ranked fused documents are passed into context for grounded synthesis.""",

'autogen_patterns.md': """# AutoGen Conversational Topologies

1. **Group Chat**: Multiple agents share a single conversation managed by a `GroupChatManager`. Next-speaker selection can be round-robin, random, manual, or dynamically selected by an LLM based on task context.
2. **Sequential Chat**: Agents interact in a predefined linear pipeline. Output/summary of one conversation feeds as input into the next.
3. **Nested Chat**: An agent triggers an internal, subordinate multi-agent dialogue to complete a specific sub-task or review before replying to the main chat, returning only a synthesized result.""",

'zep_memory.md': """# Zep: Temporal Knowledge Graphs for Agent Memory

Long-term memory engine organizing conversational history into a dynamic Temporal Knowledge Graph.
- Continuously extracts entities, relations, and episodic events into structured semantic nodes and edges.
- Models time natively: tracks when facts become valid, evolve, or expire, allowing agents to resolve contradictions.
- **Hybrid Retrieval**: Combines vector embeddings, BM25 keyword matching, and graph traversal algorithms for sub-second retrieval of relevant past context.""",

'flare_retrieval.md': """# FLARE: Forward-Looking Active REtrieval

Active RAG framework designed for factual accuracy in long-form generation.
- Dynamically decides when and what to retrieve during generation rather than once upfront.
- **Look-Ahead Mechanism**: As the model generates text, it drafts a provisional next sentence. If the draft contains low-confidence tokens, FLARE triggers retrieval.
- Extracts queries from the tentative sentence to fetch external documents and regenerates the sentence using retrieved context.""",

'crag_corrective_rag.md': """# Corrective RAG (CRAG, Yan et al. 2024)

Evaluates and rectifies retrieved documents via a lightweight retrieval evaluator:
1. **Correct**: If documents are relevant, refines them into fine-grained knowledge strips and filters noise.
2. **Incorrect**: If documents are irrelevant, discards them and executes external web search.
3. **Ambiguous**: If confidence is borderline, fuses refined internal documents with external web search results.
The synthesized knowledge is passed to the generator to eliminate hallucinations from faulty retrieval.""",

'cohere_rerank.md': """# Cross-Encoder Reranking & Cohere Rerank

Secondary stage to boost retrieval precision after bi-encoder/BM25 retrieval:
- Bi-encoders embed queries and documents separately, missing fine-grained token interactions.
- Cross-encoders evaluate query-document pairs simultaneously using full cross-attention across all tokens.
- **Production RAG**: Re-scores and filters candidates down to top-k most relevant passages before context injection, suppressing noise, reducing hallucinations, and lowering prompt token costs.""",

'reflexion_patterns.md': """# Reflexion: Verbal Reinforcement Learning (Shinn et al. 2023)

Optimizes LLM agent behavior through verbal self-reflection rather than parameter updates:
1. **Actor**: Generates actions, reasoning traces, or trajectories (ReAct/CoT).
2. **Evaluator**: Assesses outputs against environment rewards, heuristics, or ground truth.
3. **Self-Reflection**: When trajectories fail, an LLM evaluates the error signal to generate constructive natural-language critiques.
Linguistic reflections are stored in episodic memory buffer and retrieved as context in subsequent trials to self-correct plans.""",

'speculative_rag.md': """# Speculative RAG (Draft-Verify Pipelines)

Accelerates and improves RAG using a draft-and-verify paradigm:
- Retrieved documents are partitioned into distinct subsets.
- A smaller, faster "drafter" processes subsets in parallel to generate multiple diverse candidate answers.
- A larger "verifier" evaluates and validates all candidate drafts in a single pass alongside query and rationale, selecting or synthesizing the final answer.
Reduces inference latency, cuts compute costs, and mitigates distractor noise.""",

'generative_agents_memory.md': """# Generative Agents Memory Hierarchy (Park et al. 2023)

Memory structured around an observation memory stream:
- **Scoring**: Observations scored for recency, importance (1-10 scaled by LLM), and relevance.
- **Reflection**: When cumulative importance of recent events exceeds a threshold, an LLM generates salient questions, retrieves relevant memories, and synthesizes higher-level abstract takeaways.
- Reflections are written back into the memory stream as first-class objects alongside raw observations, creating recursive memory hierarchies.""",

'hyde_embeddings.md': """# HyDE: Hypothetical Document Embeddings (Gao et al. 2022)

Zero-shot retrieval technique transforming query-to-document search into document-to-document similarity:
- Instructs an LLM to generate a hypothetical answer/document to the query.
- Even if hallucinated, the synthetic document captures the semantic structure, style, and vocabulary of a relevant answer.
- An unsupervised dense encoder embeds the hypothetical document to retrieve actual candidates from the index via nearest-neighbor search, bridging query-document density gaps.""",

'openai_prompt_semantics.md': """# GPT-4o System Prompt Structure & Developer Message Semantics

**Modular Architecture**:
1. Identity & Role: Core persona and capabilities.
2. Behavioral Constraints: Tone, style, Markdown formatting, conciseness.
3. Safety & Policy: Guardrails and refusal boundaries.
4. Tool Definition: Function-calling interfaces and environment metadata.

**Developer Role Semantics**:
- Commands from the `developer` role take strict precedence over `user` messages.
- Separates immutable application directives and governance from untrusted user inputs, mitigating prompt injection.""",

'mt_bench_arena.md': """# MT-Bench & Chatbot Arena Elo Methodology (LMSYS)

- **Chatbot Arena**: Crowdsourced blind pairwise A/B testing where users vote on anonymous model outputs. Head-to-head battle outcomes are modeled via the Bradley-Terry (Elo) system using maximum likelihood estimation to produce continuous latent skill ratings.
- **MT-Bench**: Automated evaluation of multi-turn conversational capability across 80 questions across 8 domains, evaluated by strong LLM judges (GPT-4) with pairwise or 1-10 absolute grading.""",

'memgpt_letta.md': """# MemGPT / Letta Hierarchical Agent Memory

Models LLM memory after an operating system virtual memory hierarchy:
1. **In-Context (RAM)**:
   - Core Memory: Fixed-size blocks (persona, user profile) pinned in system prompt, editable via function calls.
   - Working Context: FIFO conversational message buffer.
2. **Out-of-Context (Disk)**:
   - Recall Memory: Searchable conversational history log.
   - Archival Memory: Vector database for arbitrary knowledge retrieval.
Agent autonomously pages, edits, and searches across tiers via tool calls for unbounded persistent state."""
}

written = 0
for fname, content in files.items():
    path = os.path.join(base, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    written += 1

print(f'Successfully wrote {written} snippet files.')
