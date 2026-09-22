# Parent Document Retrieval / Small-to-Big Chunking

- **Mechanism**: Splits documents into small child chunks (sentences/paragraphs) mapped to larger parent chunks or full documents.
- **Search**: Matches user queries against fine-grained child embeddings for high precision.
- **Expansion**: Replaces child chunks with full parent context prior to passing into LLM context, avoiding semantic fragmentation.