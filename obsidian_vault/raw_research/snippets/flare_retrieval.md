# FLARE: Forward-Looking Active REtrieval

Active RAG framework designed for factual accuracy in long-form generation.
- Dynamically decides when and what to retrieve during generation rather than once upfront.
- **Look-Ahead Mechanism**: As the model generates text, it drafts a provisional next sentence. If the draft contains low-confidence tokens, FLARE triggers retrieval.
- Extracts queries from the tentative sentence to fetch external documents and regenerates the sentence using retrieved context.