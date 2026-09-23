# LLaVA-NeXT AnyRes: Dynamic Resolution & Spatial Tiling (Liu et al., 2024)

## 1. Fixed-Resolution Spatial Blurring
Standard vision backbones resize images to a fixed square (e.g., $336\times 336$), causing severe Nyquist blurring on high-resolution text, small objects, and diagrams. Conversely, naive full-resolution ViT tokenization causes quadratic token sequence explosion.

## 2. AnyRes Multi-Patch Grid Partitioning
LLaVA-NeXT (Liu et al., 2024) dynamically partitions arbitrary aspect-ratio images into a grid of canonical $S \times S$ sub-patches ($S = 336$):
1. **Grid Selection:** Chooses grid $(m^*, n^*) \in \mathcal{G}$ that minimizes aspect ratio distortion:
   $$(m^*, n^*) = \operatorname{argmin}_{(m, n)} \left| \frac{W}{H} - \frac{m \cdot S}{n \cdot S} \right|$$
2. **Global Thumbnail Integration:** Concurrently processes a downsampled global overview thumbnail ($336\times 336$) to anchor macroscopic context.
3. **2D Topology Preservation:** Inserts special learned newline tokens `\n` between patch rows in the token stream, allowing the autoregressive LLM to preserve 2D coordinate layout.

## 3. Empirical Gains
Lifts DocVQA performance by **$+12.4\%$** and TextVQA by **$+8.6\%$** without requiring architectural redesign of the core vision transformer.
