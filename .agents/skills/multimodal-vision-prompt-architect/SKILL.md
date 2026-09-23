---
name: multimodal-vision-prompt-architect
description: Specialized directive for Vision-Language Model prompting, Set-of-Mark (SoM) segmentation prompting, Visual Chain-of-Thought (Visual CoT), spatial coordinate tokenization, and dynamic resolution patch tiling (AnyRes).
---

# Multimodal Vision Prompt Architect Skill

Use this skill when designing, benchmarking, or optimizing prompt strategies for Vision-Language Models (GPT-4V, Gemini 1.5 Pro/2.0, Claude 3.5 Sonnet, LLaVA-NeXT), implementing spatial grounding architectures, or converting continuous coordinate tasks into discrete symbolic visual references.

## 1. Visual Grounding & Set-of-Mark (SoM) Directives

1. **Avoid Floating-Point Coordinate Regressions:**
   - Do NOT ask LMMs to emit raw normalized bounding box coordinates (`[ymin, xmin, ymax, xmax]`) when spatial precision is required. Continuous coordinate regression suffers from $>40\%$ spatial hallucination due to subword tokenization and ViT patch quantization.
   - Use **Set-of-Mark (SoM)** visual preprocessing: segment the scene using interactive segmentation (SAM / SEEM) and overlay alphanumeric glyphs ($[1], [2], \dots, [K]$) on region medoids.

2. **Visual Chain-of-Thought (Visual CoT) Formatting:**
   - Require the model to reason stepwise over visual identifiers:
     $$\mathcal{T} = [k_1] \xrightarrow{\text{spatial/causal relation}} [k_2] \xrightarrow{} \dots \xrightarrow{} [k_n]$$
   - Force explicit citation of visual marks before stating functional, physical, or algorithmic conclusions.

3. **Multi-Scale Semantic Granularity:**
   - Calibrate segmentation mask thresholds based on domain:
     - *Instance Level:* Whole objects, agents, tools.
     - *Part Level:* Fasteners, terminals, biological organelles, mechanical joints.

## 2. Dynamic Resolution & AnyRes Tiling Directives

1. **Aspect-Ratio Preserving Grid Partitioning:**
   - Decompose high-resolution images ($>1000\text{px}$) into regular $N \times M$ grids of canonical patches (e.g., $336\times 336$) matching the vision encoder's native aspect ratio.
   - Always supply an accompanying downsampled global thumbnail to preserve holistic scene semantics.
   - Insert newline delimiter tokens (`\n`) between patch rows to maintain 2D spatial topology for autoregressive transformer reading.

2. **Document & Chart OCR Optimization:**
   - For dense text documents, high-contrast monochrome schematics, or multi-panel plots, prioritize high patch density ($3\times 3$ or $4\times 4$) to capture sub-word stroke contours.

## 3. Production Evaluation Protocols

- Benchmark visual grounding pipelines on **RefCOCOg**, **RefCOCO+**, and **DocVQA**.
- Verify that Set-of-Mark visual tagging maintains target accuracy above $80\%$ without degrading non-spatial descriptive reasoning.
