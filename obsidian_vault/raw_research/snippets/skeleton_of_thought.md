# Skeleton-of-Thought (SoT) (Ning et al., ICLR 2024)

- **Skeleton Drafting**: Generates a concise outline/bullet points instead of full sequential text.
- **Point-by-Point Parallel Generation**: Each skeleton point is elaborated simultaneously and independently via concurrent API requests or batched GPU decoding.
- **Speedup**: Up to 2.39x average speedup across question types; peak acceleration 2.69x-2.88x without quality loss.
- **Latency**: Exploits GPU batch parallelization during decoding to cut memory-bound token latency.