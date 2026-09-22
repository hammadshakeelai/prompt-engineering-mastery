# Visual Chain-of-Thought, Set-of-Mark (SoM) & Spatial Grounding (2024–2026)

## 1. The Spatial Disconnect in Frontier VLMs
While modern Vision-Language Models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5/2.0 Pro) excel at holistic scene classification, they exhibit severe spatial hallucination when identifying fine-grained spatial coordinates or distinguishing multiple instances of identical objects.

## 2. Set-of-Mark (SoM) Prompting (Yang et al. CVPR 2024)
Introduced by Microsoft Research, **Set-of-Mark (SoM)** decouples visual perception from spatial reference:
1. **Segmentation Pre-Pass:** An off-the-shelf segmentation foundation model (SAM or SEEM) partitions the input image into discrete semantic masks.
2. **Visual Tagging:** Overlays numerical or alphabetical markers directly onto the pixel canvas:
   $$I_{\text{tagged}} = \mathcal{M}(I, \{\text{mask}_i, \text{label}_i\}_{i=1}^K)$$
3. **Symbolic Grounding:** The prompt references regions via discrete alphanumeric IDs (`"What is the function of component [14] relative to [8]?"`).
4. **Empirical Gain:** Eliminates spatial ambiguity, boosting GPT-4V grounding performance by $+23\%$ on RefCOCOg and UI navigation tasks.

## 3. Visual Chain-of-Thought & The Visual Sketchpad (Hu et al. 2024)
Extends autoregressive chain-of-thought to intermediate visual reasoning artifacts:
- **Executable Sketchpad:** When solving complex geometric, architectural, or charting problems, the VLM emits executable Python code (PIL/OpenCV) that alters the image (e.g., drawing auxiliary proof lines, cropping bounding boxes, zooming in).
- **Interleaved Vision-Language Verification:** The rendered canvas is fed back into the VLM's next reasoning turn, creating a visual feedback loop that improves multi-step geometric problem solving by up to $35\%$.
