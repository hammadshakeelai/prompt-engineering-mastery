# Generative Agents Memory Hierarchy (Park et al. 2023)

Memory structured around an observation memory stream:
- **Scoring**: Observations scored for recency, importance (1-10 scaled by LLM), and relevance.
- **Reflection**: When cumulative importance of recent events exceeds a threshold, an LLM generates salient questions, retrieves relevant memories, and synthesizes higher-level abstract takeaways.
- Reflections are written back into the memory stream as first-class objects alongside raw observations, creating recursive memory hierarchies.