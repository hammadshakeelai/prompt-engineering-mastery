# Set-of-Mark (SoM) Visual Grounding & Spatial CoT (Yang et al., CVPR 2024)

## 1. Spatial Grounding Breakdown in Vision-Language Models
Frontier Large Multimodal Models (LMMs like GPT-4V, Gemini 1.5 Pro) fail on fine-grained spatial grounding and referring expression comprehension when prompted with raw bounding box coordinates (`[ymin, xmin, ymax, xmax]`), exhibiting $>40\%$ spatial hallucination.

## 2. The Set-of-Mark (SoM) Pipeline
Yang et al. (*SoM*, Microsoft / CVPR 2024 / arXiv:2310.11441) transform continuous coordinate prediction into discrete symbolic visual reasoning:
1. **Interactive Segmentation:** Generates semantic region masks $\{M_1, \dots, M_K\}$ using SAM or Semantic-SAM.
2. **Visual Mark Superimposition:** Overlays high-contrast alphanumeric glyphs ($[1], [2], \dots, [K]$) and colored boundaries onto region medoids.
3. **Symbolic Language Grounding:** Prompts the LMM to reason directly over marked symbols $[k]$.

## 3. Visual Chain-of-Thought (Visual CoT)
Enables multi-step visual derivations where reasoning steps trace through mark identifiers ($[2] \to [5] \to [8]$) to establish spatial containment, functional hierarchy, and object relations.
- **RefCOCOg Benchmark:** Zero-shot GPT-4V with SoM achieves **$84.2\%$ accuracy**, surpassing fully fine-tuned specialized models without parameter retraining.
