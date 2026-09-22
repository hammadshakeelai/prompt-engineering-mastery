# Mixture of Experts (MoE) Architecture

## Sparse Activation
MoE replaces dense FFNs with parallel expert subnetworks + a learned gating router.
- Top-k Gating: Each token dispatched to top-2 experts (Mixtral). 46.7B params but ~12.9B activated per token.
- Fine-Grained (DeepSeek-MoE): 64 routed + 2 shared experts. Shared experts process every token; routed capture specialized knowledge.

## Load Balancing
- Auxiliary Loss: Penalizes non-uniform token-to-expert distribution (Switch Transformer, Mixtral).
- Loss-Free Balancing (DeepSeek-V3): Adjusts per-expert bias terms based on workload without degrading LM performance.
- Capacity Factor: Sets max token quotas per expert buffer; excess tokens bypassed or dropped.

## Prompt Design Implications
- Routing Sensitivity: Small phrasing shifts can flip top-k dispatch decisions.
- Domain Steering: Domain keywords / role framing shifts activation vectors into specialized expert subspaces.
- Structural Consistency: Uniform templates ensure stable routing paths.