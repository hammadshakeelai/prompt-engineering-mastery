# Grammar-Constrained Speculative Decoding (GSD & Speculative Grammar Trees)

## 1. Breakdown of Unconstrained Speculative Drafting
Under standard speculative decoding (Leviathan et al., 2023), small draft model $M_d$ generates $\gamma$ tokens verified in parallel by target model $M_t$.
- **The Structured Output Bottleneck:** In JSON/YAML or code generation, syntax requires strict adherence to grammar $\mathcal{G}$.
- When $M_d$ drafts without grammar constraints, it samples invalid syntactic tokens with high frequency. Because the target verifier masks invalid tokens, draft acceptance rate collapses ($\alpha \to 0$), nullifying speculative speedups.

## 2. Grammar-Synchronized Speculative Protocols
In grammar-constrained speculative architectures (e.g., GSD, Lookahead-then-Verify):
1. **Coupled Logit Masking:**
   At draft step $k$, logits are filtered through automata state $s_k$:
   $$P_d^\mathcal{G}(x_k \mid x_{<k}) = \frac{P_d(x_k \mid x_{<k}) \cdot \mathbb{I}[x_k \in \mathcal{V}_{\text{valid}}(s_k)]}{\sum_{v \in \mathcal{V}_{\text{valid}}(s_k)} P_d(v \mid x_{<k})}$$
2. **Synchronized Rejection Sampling:**
   Verification applies the identical constraint:
   $$\alpha_\mathcal{G}(x_k) = \min\left(1, \frac{P_t^\mathcal{G}(x_k \mid x_{<k})}{P_d^\mathcal{G}(x_k \mid x_{<k})}\right)$$
   guaranteeing mathematical equivalence to target-only constrained sampling.

## 3. Fast-Forwarding & Tree Verification
1. **Deterministic Path Fast-Forwarding:**
   When $|\mathcal{V}_{\text{valid}}(s_k)| = 1$ (e.g., structural delimiters `": "`, `", "`):
   - The token is emitted deterministically without invoking neural network draft passes.
2. **Speculative Grammar Trees:**
   When multiple tokens are valid, $M_d$ spans a speculative tree $\mathcal{T}$. Target model $M_t$ verifies the full tree in a single forward pass using custom causal tree attention masks:
   $$M_{i, j} = \begin{cases} 0 & \text{if } j \text{ is an ancestor of } i \text{ in } \mathcal{T} \\ -\infty & \text{otherwise} \end{cases}$$
3. **Performance:** Acceptance rates exceed $90\%$ across structured tasks, yielding $3.8\times\text{--}6.2\times$ wall-clock speedups over autoregressive baselines.
