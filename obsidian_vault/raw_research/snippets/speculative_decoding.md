# Speculative Decoding for LLM Inference Speed

## Architecture
- Drafting: Small draft mechanism proposes K candidate tokens autoregressively.
- Verification: Large target model validates all K candidates in ONE parallel forward pass.

## Acceptance Mechanics
Token x accepted with probability min(1, p(x)/q(x)) where p=target, q=draft.
First rejection triggers corrected sample from max(0, p(x)-q(x)). Output is lossless.

## Advanced Architectures
- Medusa: Multiple MLP heads atop frozen target backbone. Each head k predicts token t+k. Tree attention verifies simultaneously.
- EAGLE/EAGLE-2: Speculates at feature level (hidden states). Single transformer decoder layer predicts future features. Acceptance >80%.

## Speed Gains
- Typical 2-4x wall-clock speedup in memory-bound regimes (batch=1).
- Structured outputs: Deterministic syntax (brackets, keys) pushes acceptance near ~100%, enabling >4x speedup.