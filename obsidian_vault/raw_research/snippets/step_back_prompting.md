# Step-Back Prompting (Zheng et al., 2023)

1. **Abstraction**: Model generates and answers a broader, higher-level "step-back question" stripping away instance-specific noise.
2. **Reasoning**: Solves original question grounded in the retrieved first principles/governing concepts.
3. **Benchmarks**: +7% MMLU Physics, +11% MMLU Chemistry, up to +36% on TimeQA/MuSiQue across PaLM-2L, GPT-4, and LLaMA-2-70B.