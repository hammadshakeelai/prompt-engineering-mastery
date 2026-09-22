# Prompt Caching Economics: Anthropic, OpenAI, DeepSeek

- **Anthropic**: Explicit breakpoint markers (min 1,024 tokens). 25% write surcharge, 90% discount on cache hits with 5-minute refreshable TTL.
- **OpenAI**: Automatic caching on prefixes >= 1,024 tokens. No write surcharge, 50% discount on cache hits.
- **DeepSeek**: Automatic caching at 64-token granularity. No write surcharge, up to ~90% discount on hits, delivering the lowest absolute inference cost.