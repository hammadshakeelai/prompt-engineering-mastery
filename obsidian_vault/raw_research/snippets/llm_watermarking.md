# LLM Output Watermarking

## Red-Green Token Watermarking (Kirchenbauer et al., 2023)
Partitions vocabulary into green list G and red list R via pseudo-random hash of preceding tokens.
Additive bias delta > 0 injected into green token logits before softmax, biasing sampling toward G.

## Semantic Watermarking
Hash-based token watermarks break under lexical edits.
SemStamp ties partitioning to semantic embeddings / sentence-level latent spaces, maintaining robustness across paraphrase boundaries.

## Detection: Z-Score Statistical Test
z = (|s|_G - gamma*T) / sqrt(T*gamma*(1-gamma))
where |s|_G = observed green tokens across length T.
Standard threshold z >= 4 (p < 3.2e-5) provides bounded false positive rates.

## Limitations
- Low-Entropy: Deterministic outputs (code, math) suffer quality loss or insufficient green tokens.
- Evasion: Vulnerable to heavy paraphrasing, translation round-trips, token insertion/deletion.
- Length: Short texts (T < 50 tokens) lack statistical power.

## Implications for Attribution
Enables zero-bit / multi-bit provenance tracking for copyright, disinformation, academic integrity.
Open-weights models can bypass; key compromise enables spoofing.