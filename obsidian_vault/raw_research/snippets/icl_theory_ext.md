# In-Context Learning (ICL) Theory and Exemplar Mechanics

## 1. Implicit Bayesian Inference (Xie et al., 2022)
ICL framed as implicit Bayesian inference over pre-trained latent concepts.
Pre-training on long-range coherent text forces Transformer to model a mixture of latent task variables.
During inference, few-shot demos condition the posterior P(theta | x_1:k), activating the relevant pre-existing latent task concept without weight updates.

## 2. Implicit Gradient Descent in Forward Pass (Akyurek et al., 2022)
Transformers mechanistically implement gradient descent and ridge regression within their forward pass.
- Linear attention layers function as meta-optimizers.
- Self-attention computes query-key moments matching linear weight updates.
- Activations store an implicit parameter vector w_hat, updated layer-by-layer using in-context pairs as an ephemeral training set.

## 3. What Makes an Effective Few-Shot Exemplar (Min et al., 2022)
Exemplars drive task retrieval, not raw parameter learning:
- Format: Strict template consistency anchors induction heads to parse delimiters and output syntax. Format clarity often outweighs ground-truth label correctness.
- Diversity: High semantic and syntactic diversity (MMR selection) covers distinct decision boundaries and input manifold.
- Label Distribution: Balanced class distributions prevent model skew. Distribution defines the valid label space and calibrates output priors P(Y).