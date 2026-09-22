# Multimodal Prompting for Vision-Language Models (VLMs)

## 1. Set-of-Mark (SoM) Prompting
Set-of-Mark (SoM) prompting is a visual prompting technique designed to enhance the fine-grained visual grounding capabilities of Large Multimodal Models (LMMs). Standard VLMs often struggle to pinpoint or distinctly refer to specific objects in complex scenes.
- **How it works:** An off-the-shelf interactive segmentation model (like SAM or GroundingDINO) is first used to partition the image into distinct regions. These regions are overlaid with visual marks (e.g., alphanumeric labels, numbers, or bounding boxes). The VLM is then prompted using the marked image (e.g., "What is the object at mark 5?").
- **Benefits:** It provides the model with a "spatial vocabulary," reducing ambiguity and enabling precise object identification and reasoning without the need for extensive model fine-tuning. This is especially useful for visual agents interacting with user interfaces or complex environments.

## 2. Spatial Bounding Box Grounding
Spatial bounding box grounding bridges high-level semantic understanding with exact spatial perception by mapping natural language descriptions to localized bounding boxes in an image.
- **Challenges:** Traditional VLMs generate coordinates as long sequences of numeric tokens (e.g., `[x1, y1, x2, y2]`), which suffers from inefficiency, formatting errors, and heavy computation due to autoregressive decoding.
- **Innovations:** Recent methods like *LocateAnything* use parallel box decoding to treat boxes as atomic units instead of sequential text tokens. Other frameworks introduce plugin guidance modules or region-aware architectures (e.g., *VLX-Seek*) that retrieve region tokens rather than predicting coordinates, aligning better with how LLMs process entities.
- **3D Evolution:** Newer models are pushing beyond 2D boxes into 3D spatial reasoning via monocular depth estimation and scene graphs, vital for robotics and embodied AI.

## 3. Image Tiling Strategies for High-Resolution Inputs
Tiling helps VLMs overcome the "resolution curse," where fine details are lost when high-resolution images are downscaled to fit standard encoder constraints (e.g., 224x224 or 336x336 pixels).
- **Tiling & Global Context:** Images are split into grids (e.g., 2x2 or 3x3) processed independently by the vision encoder. To retain semantic coherence across the entire image, a downsampled "global view" is often processed alongside the high-res tiles.
- **Dynamic Resolution:** More advanced models use dynamic, aspect-ratio-aware cropping rather than fixed grids. Some use attention-guided token pruning to focus compute only on information-dense regions.
- **Trade-offs:** Token explosion is a major challenge. Because self-attention scales quadratically, multiplying the number of patches sharply increases latency and memory usage. "Semantic stitching" techniques are sometimes used to handle objects sliced at tile boundaries.

## 4. OCR Layout Prompting
VLMs are replacing multi-stage OCR pipelines with end-to-end, layout-aware extraction by simultaneously processing text and visual layout structure.
- **Structure-Oriented Prompts:** Prompting the VLM to return structured formats (JSON, Key-Value, Markdown) forces the model to respect visual hierarchies rather than producing a flat text stream.
- **Chain-of-Thought (CoT):** Asking the model to "reason" over the layout (e.g., "Identify the table headers first, then extract the row data") greatly improves accuracy for visually complex documents.
- **Input Ordering & Few-Shot:** Providing image-to-structure examples in the prompt and feeding the image before text instructions leads to marked performance gains.
- **Hybrid Systems:** While VLMs handle structure and complex visual components, traditional OCR is still frequently paired with VLMs in hybrid pipelines to minimize hallucination and API costs.

## 5. Chain-of-Thought (CoT) for Visual Reasoning
CoT in VLMs combats hallucinations and failures in multi-step visual inference by forcing the model to decompose tasks into observable, intermediate steps before outputting an answer.
- **Visual Chain-of-Thought (VCoT):** This approach integrates explicit visual states (like rendering bounding boxes or generating synthesized diagrams) into the reasoning sequence, bridging text rationale with visual grounding.
- **Continuous Visual Tokens:** Advanced models use "Chain-of-Visual-Thought" (CoVT) to reason directly in visual latent spaces, employing continuous tokens that encode geometry and depth instead of relying solely on textual rationales.
- **Training Advances:** High-quality CoT reasoning is currently being distilled from frontier models (like GPT-4o) using techniques such as Direct Preference Optimization (DPO) and reinforcement learning to penalize logic gaps and ground the reasoning strictly in the image content.
