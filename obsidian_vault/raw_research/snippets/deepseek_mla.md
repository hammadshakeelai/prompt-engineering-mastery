# DeepSeek Multi-Head Latent Attention (MLA)

- **Latent Compression**: Projects hidden states into a compact low-dimensional latent vector via shared down-projection matrices, cutting KV cache memory footprint by up to 93%.
- **Decoupled RoPE**: Only the compressed latent vector and a decoupled positional key vector for RoPE are stored in the cache.
- **Matrix Absorption**: Up-projection matrices are absorbed into query projections during inference, avoiding explicit decompression while retaining full MHA expressive power.