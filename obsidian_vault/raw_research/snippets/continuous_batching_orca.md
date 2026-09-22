# Continuous Batching (Orca) vs. Static Batching

- **Static Batching**: Sequences batched together until all finish; requires padding and causes idle GPU stalls.
- **Continuous Batching (Orca)**: Iteration-level scheduling. Finished requests are evicted and new requests inserted at every decoding step without pausing ongoing generations, eliminating padding waste and maximizing throughput.