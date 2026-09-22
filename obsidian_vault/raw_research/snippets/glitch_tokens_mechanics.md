# Glitch Tokens & Embedding Geometry (SolidGoldMagikarp, petertodd)

- **Root Cause**: BPE tokenizers assign dedicated tokens to frequent strings in web scrapes (e.g., Reddit usernames) that were subsequently filtered out or absent during model pretraining and RLHF.
- **Embedding Space Anomaly**: Their embedding vectors received zero or near-zero gradient updates, leaving them ungrounded in latent space.
- **Detection**: Discovered via k-means clustering on token embeddings where glitch tokens appear as extreme geometric outliers. Prompting them triggers model instability, hallucinatory repetitions, or evasion.