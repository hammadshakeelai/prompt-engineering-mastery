# Representation Engineering (RepE, Zou et al. 2023)

- **Concept Vectors**: Isolates high-level directions in activation space (honesty, deception, emotion) via contrastive prompt pairs and PCA / difference-in-means across intermediate layers.
- **Reading**: Probes internal states to monitor latent deception or bias during generation.
- **Control**: Steers model behavior at inference by adding or subtracting scaled concept vectors from hidden activations without fine-tuning.