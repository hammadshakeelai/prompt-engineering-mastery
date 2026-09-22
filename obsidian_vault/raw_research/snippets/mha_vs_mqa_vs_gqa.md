# Attention Variants: MHA vs. MQA vs. GQA

- **Multi-Head Attention (MHA)**: Independent Query, Key, and Value heads ($H$ Q, $H$ K, $H$ V). Maximum representational capacity, highest KV cache memory and bandwidth costs.
- **Multi-Query Attention (MQA)**: $H$ Query heads share a single Key and Value head ($H$ Q, 1 K, 1 V). Drastically shrinks KV cache footprint, risks minor quality degradation.
- **Grouped-Query Attention (GQA)**: Divides $H$ Query heads into $G$ groups, each group sharing one Key and Value head ($H$ Q, $G$ K, $G$ V). Delivers MQA-level speed with near-MHA accuracy.