# Function Vectors & Task Representation Geometry (Todd et al., ICLR 2024)

## 1. Internal Representation of In-Context Learning (ICL)
When transformers perform few-shot in-context learning:
$$\mathcal{P} = \{ (x_1, y_1), \dots, (x_k, y_k), x_* \}$$
how is the abstract operator $f: X \to Y$ stored?
Todd et al. (ICLR 2024) proved that transformers distill abstract input-output mappings into a single, compact **Function Vector (FV)** localized in intermediate attention heads.

## 2. Causal Mediation & Extraction Protocol
1. **Task Heads ($H_{\text{task}}$):**
   Using activation patching across individual attention heads, causal mediation analysis reveals that only a small subset of heads ($<15$ heads in middle layers, e.g., layers $14\text{--}24$) mediate task execution at the query delimiter token.
2. **Extraction Formula:**
   The function vector $\mathbf{v}_f \in \mathbb{R}^d$ is computed by taking the expectation of task head output projections across a distribution of demonstration prompts $\mathcal{D}_f$:
   $$\mathbf{v}_f = \frac{1}{|\mathcal{D}_f|} \sum_{p \in \mathcal{D}_f} \sum_{(l, h) \in H_{\text{task}}} W_O^{(l, h)} \text{Attn}^{(l, h)}(p)_{-1}$$
   where $W_O^{(l, h)}$ is the output projection matrix.

## 3. Zero-Shot Steering & Compositional Algebra
- **Promptless Zero-Shot Activation:** Adding $\beta \cdot \mathbf{v}_f$ to the residual stream of an unprompted input $x_*$ triggers immediate task execution (translation, antonyms, capital retrieval) without text exemplars.
- **Template Invariance:** Vectors extracted from `"Input: X -> Output: Y"` successfully drive generation in completely distinct syntax (`"The antonym of X is..."`), proving abstraction beyond lexical formatting.
- **Linear Compositionality:** Multi-task behaviors can be composed via vector addition in activation space ($\mathbf{v}_{\text{comp}} = \alpha \mathbf{v}_{f_1} + \gamma \mathbf{v}_{f_2}$), demonstrating that algorithmic subroutines occupy linear subspaces.
