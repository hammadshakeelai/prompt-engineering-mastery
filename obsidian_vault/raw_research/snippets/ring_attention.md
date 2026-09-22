# RingAttention for Million-Token Contexts

- **Ring Topology**: Shards query, key, and value tensors across multiple GPUs in a logical ring rather than gathering full sequences.
- **Communication Overlapping**: Overlaps blockwise attention compute with peer-to-peer asynchronous transmission of key-value blocks.
- **Linear Scaling**: Memory requirements scale with local chunk size $O(N/P)$ rather than total length $N$, enabling million-token context training and inference.