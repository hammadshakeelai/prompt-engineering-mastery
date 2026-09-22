# TransformerLens: Residual Stream Interventions & Causal Tracing

- **Residual Stream Interventions**: Hooks into `resid_pre` and `resid_post` to read, steer, ablate, or overwrite activation vectors at specific layers and token positions.
- **Activation Patching**: Swaps activations between clean and corrupted forward passes to establish causal links, measuring sufficiency (denoising) and necessity (noising) to isolate functional circuits.