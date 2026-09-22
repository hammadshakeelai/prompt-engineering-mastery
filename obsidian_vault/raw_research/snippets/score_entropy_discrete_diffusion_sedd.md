# Score Entropy Discrete Diffusion (SEDD) (Lou et al., ICML 2024 Best Paper)

## 1. Limitations of Autoregressive Generation
Autoregressive language models factorize joint sequence probability strictly from left to right:
$$P(x_{1:T}) = \prod_{t=1}^T P(x_t \mid x_{<t})$$
- **Unidirectional Planning Trap:** Early tokens cannot be updated or revised after generation; mistakes in early steps permanently corrupt the downstream trajectory.
- **Infilling Rigidity:** Generating text conditioned on arbitrary prefix and suffix boundaries requires complex scaffolding or re-prompting.

## 2. Discrete Score Matching & Score Entropy
In continuous diffusion, the score function $\nabla_x \log p_t(x)$ estimates gradient directions towards data modes. In discrete token spaces $\mathcal{V}^T$, gradients are undefined.
Lou et al. (Stanford / ICML 2024 Best Paper) extend score matching to discrete spaces via concrete probability ratios:
$$s_\theta(x, t)_{i, y} \approx \frac{p_t(x \setminus \{x_i\} \cup \{y\})}{p_t(x)}$$
- **Score Entropy Loss:**
  $$\mathcal{L}_{\text{SEDD}}(\theta) = \mathbb{E}_{t, x_0, x_t}\left[ \sum_{i=1}^T \sum_{y \in \mathcal{V}} w_t(y) \left( s_\theta(x_t, t)_{i, y} - \log s_\theta(x_t, t)_{i, y} \right) \right]$$
  Directly learns the ratio of transition probabilities under a continuous-time Markov jump process without requiring categorical relaxation or Gumbel-Softmax approximations.

## 3. Capabilities & Non-Autoregressive Steering
- **Bidirectional Infilling:** Naturally conditions on arbitrary prompt masks (prefix, suffix, or interstitial text), iteratively denoising masked tokens simultaneously.
- **Controllable Compute-Quality Tradeoff:** Sampling steps $N_{\text{steps}}$ can be dynamically adjusted at inference time ($16$ to $1024$ steps) without changing model weights.
- **Performance:** Outperforms prior discrete diffusion models (D3PM, Bit-Diffusion) and matches or exceeds GPT-2 perplexity baselines on standard language modeling benchmarks.
