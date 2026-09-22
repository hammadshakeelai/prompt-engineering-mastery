# H2O: Heavy-Hitter Oracle & Submodular KV Eviction (Zhang et al. NeurIPS 2023)

## 1. Attention Heavy-Hitter ($H_2$) Power Law
In long-context LLM inference, memory bandwidth and VRAM are bounded by KV cache scaling $\mathcal{O}(B \cdot L \cdot T \cdot d)$.
Zhang et al. empirically proved that attention mass concentrates exponentially into a small fraction ($\le 20\%$) of tokens called **Heavy Hitters ($H_2$)**:
$$S_j = \sum_{t=j}^T A_{t, j}$$
- **Token Taxonomy:**
  1. *Attention Sinks:* Initial $2\text{--}4$ prompt tokens serving as softmax score anchors.
  2. *Syntactic Separators:* Punctuation and formatting boundaries structuring clauses.
  3. *Semantic Anchors:* Core entities repeatedly attended to during generation.

## 2. Dynamic Submodular Optimization
Cache retention is formalized as a constrained submodular set function optimization problem:
$$\max_{S \subseteq [T], |S| \le C} F(S) = \sum_{j \in S} S_j$$
- **Greedy Eviction:** When current cache size exceeds limit $C$, evict the non-local token with the minimum accumulated attention:
  $$j^* = \arg\min_{j \in H_2 \setminus W_{\text{recent}}} S_j$$
- **Theoretical Guarantee:** Greedy eviction guarantees a $(1 - 1/e) \approx 63.2\%$ approximation bound to the omniscient offline oracle.
- **Speedup:** Achieves up to $29\times$ throughput expansion and $1.9\times$ latency reduction with negligible perplexity penalty.

## 3. Layer Asymmetry (SnapKV & PyramidKV)
Uniform layer budgets waste memory in higher layers. Pyramidal scheduling progressively tightens KV budgets across network depth:
$$C_l = C_{\text{base}} \cdot \left(1 - \alpha \frac{l}{L}\right)$$
preserving diffuse attention in shallow layers while enforcing extreme sparsity in deep layers.
