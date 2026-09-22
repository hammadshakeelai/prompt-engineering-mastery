# Prefix-Tuning (Li & Liang, 2021)

- **Virtual Prefix Vectors**: Freezes pretrained weights and prepends continuous, learnable task-specific vectors to keys and values across all attention layers.
- **Reparameterization**: Uses temporary MLP reparameterization during training for gradient stability.
- **Parameter Efficiency**: Tunes only ~0.1% parameters while achieving full fine-tuning performance.