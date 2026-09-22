# Attention Mechanisms and Positional Encodings for Prompting

## 1. Dot-Product Attention and Token Saliency
A = softmax(QK^T / sqrt(d_k)) * V
Query-key dot products measure semantic compatibility. Softmax exponentially amplifies top-scoring pairs.
Tokens whose keys align strongly with generation queries capture attention mass; noisy tokens suffer dilution.

## 2. Absolute vs. Relative Encodings
- Absolute (Sinusoidal/Learned): Fixed index coordinates, fail to extrapolate beyond pre-trained context window.
- Relative (ALiBi, T5 bias): Inject distance offsets directly into attention logits, generalizing robustly.
- RoPE: Rotates Q and K in complex 2D sub-spaces. Preserves vector norms, naturally encodes relative distance.
  Permits context window expansion via frequency scaling (YaRN, NTK).

## 3. Practical Prompt Structure Implications
- Boundary Placement (Sandwiching): Anchor core system rules at token 0; place queries/schema at final tokens.
- Delimiters as Attention Anchors: XML tags (<context>, <doc>) create distinct key vectors query heads latch onto.
- Salience vs. Distractors: Softmax is zero-sum; extraneous tokens siphon attention away from critical instructions.
- Attention Sinks: Token 0 absorbs excess attention mass in many models (StreamingLLM finding).