import os

base = r'c:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\raw_research\snippets'
os.makedirs(base, exist_ok=True)

files = {
'flashattention_2_vs_3.md': """# FlashAttention-2 vs. FlashAttention-3 Architectural Optimizations

- **Hardware Targeting**: FA-2 optimized for Ampere via sequence parallelization. FA-3 targets Hopper (H100) Tensor Memory Accelerator (TMA) and hardware barriers (mbarrier) for asynchronous data movement between GMEM and SMEM.
- **Warp Specialization**: FA-3 separates warps into dedicated producer (TMA loading) and consumer (Tensor Core MMA) warps, removing FA-2 cross-warp synchronization overhead.
- **Interleaved GEMM-Softmax**: Overlaps Tensor Core matmuls with ALU softmax operations (ping-pong pipelining).
- **FP8 Support**: Native FP8 execution with block scaling doubles compute throughput over FA-2 FP16 baseline.""",

'paged_attention_vllm.md': """# PagedAttention & vLLM Prefix Caching

- **Virtual Memory Architecture**: Partitions KV cache into non-contiguous fixed-size blocks (pages), eliminating external fragmentation.
- **Prefix Caching**: Identical prefix tokens (system prompts, few-shot examples) map to shared physical memory blocks through block tables and reference counting.
- **Copy-on-Write**: Requests reuse existing physical pages, allocating new pages only when tokens diverge during generation, saving TTFT latency and memory.""",

'mha_vs_mqa_vs_gqa.md': """# Attention Variants: MHA vs. MQA vs. GQA

- **Multi-Head Attention (MHA)**: Independent Query, Key, and Value heads ($H$ Q, $H$ K, $H$ V). Maximum representational capacity, highest KV cache memory and bandwidth costs.
- **Multi-Query Attention (MQA)**: $H$ Query heads share a single Key and Value head ($H$ Q, 1 K, 1 V). Drastically shrinks KV cache footprint, risks minor quality degradation.
- **Grouped-Query Attention (GQA)**: Divides $H$ Query heads into $G$ groups, each group sharing one Key and Value head ($H$ Q, $G$ K, $G$ V). Delivers MQA-level speed with near-MHA accuracy.""",

'deepseek_mla.md': """# DeepSeek Multi-Head Latent Attention (MLA)

- **Latent Compression**: Projects hidden states into a compact low-dimensional latent vector via shared down-projection matrices, cutting KV cache memory footprint by up to 93%.
- **Decoupled RoPE**: Only the compressed latent vector and a decoupled positional key vector for RoPE are stored in the cache.
- **Matrix Absorption**: Up-projection matrices are absorbed into query projections during inference, avoiding explicit decompression while retaining full MHA expressive power.""",

'streaming_llm_sinks.md': """# StreamingLLM: Attention Sinks & Sliding Windows

- **Attention Sinks**: LLMs dedicate disproportionately large attention weights to initial tokens (token 0) to absorb excess attention mass, regardless of semantic content.
- **Mechanism**: Preserves initial attention sink tokens (first 4 tokens) alongside the sliding window of latest tokens in KV cache.
- **Impact**: Restores softmax stability, avoids perplexity spikes, and enables infinite-length text streaming with fixed memory footprint without fine-tuning.""",

'ring_attention.md': """# RingAttention for Million-Token Contexts

- **Ring Topology**: Shards query, key, and value tensors across multiple GPUs in a logical ring rather than gathering full sequences.
- **Communication Overlapping**: Overlaps blockwise attention compute with peer-to-peer asynchronous transmission of key-value blocks.
- **Linear Scaling**: Memory requirements scale with local chunk size $O(N/P)$ rather than total length $N$, enabling million-token context training and inference.""",

'positional_encodings_extrapolation.md': """# Context Extrapolation: RoPE vs. YaRN vs. ALiBi

- **RoPE**: Encodes position by rotating Q and K vectors in 2D subspaces. Struggles on unseen lengths due to out-of-distribution rotation angles.
- **ALiBi**: Discards vector position embeddings; injects linear distance penalties ($-m \\cdot |i - j|$) directly into attention logits, providing zero-shot length extrapolation.
- **YaRN**: Non-uniform frequency scaling for RoPE: interpolates low frequencies, keeps high frequencies unscaled, and rescales attention logits to prevent entropy dilution.""",

'continuous_batching_orca.md': """# Continuous Batching (Orca) vs. Static Batching

- **Static Batching**: Sequences batched together until all finish; requires padding and causes idle GPU stalls.
- **Continuous Batching (Orca)**: Iteration-level scheduling. Finished requests are evicted and new requests inserted at every decoding step without pausing ongoing generations, eliminating padding waste and maximizing throughput.""",

'prompt_caching_economics.md': """# Prompt Caching Economics: Anthropic, OpenAI, DeepSeek

- **Anthropic**: Explicit breakpoint markers (min 1,024 tokens). 25% write surcharge, 90% discount on cache hits with 5-minute refreshable TTL.
- **OpenAI**: Automatic caching on prefixes >= 1,024 tokens. No write surcharge, 50% discount on cache hits.
- **DeepSeek**: Automatic caching at 64-token granularity. No write surcharge, up to ~90% discount on hits, delivering the lowest absolute inference cost.""",

'speculative_decoding_dynamics.md': """# Speculative Decoding: Acceptance Rates & Draft Sizing

- **Acceptance Dynamics**: Probability governed by modified rejection sampling $\\min(1, p_T/p_D)$. Higher in low-entropy regimes (code, grammar) and lower under high temperature.
- **Draft Model Sizing**: Optimal draft model is 10-50x smaller (~5-15% of target parameter count). Net speedup requires draft latency $t_D \\ll t_T$ while sustaining acceptance rates $\\ge 0.6-0.7$.""",

'medusa_speculative.md': """# Medusa: Draft-Free Multi-Head Speculative Decoding

- **Multi-Head Drafting**: Appends multiple lightweight MLP heads to the target LLM's final hidden state, predicting future tokens (t+1, t+2) concurrently without a draft model.
- **Tree Verification**: Candidates structured into a tree verified in a single forward pass via tree-structured attention, emitting multiple tokens per step for 2x-3x speedup.""",

'eagle_feature_speculative.md': """# EAGLE: Feature-Level Speculative Decoding

- **Feature-Space Speculation**: Uses a single-layer Transformer decoder to autoregressively predict second-to-top hidden features rather than discrete tokens.
- **Mitigating Compounding Error**: Continuous feature prediction reduces distribution shift, boosting draft acceptance rates (>80%) and yielding 2-3x lossless wall-clock acceleration.""",

'prefix_tuning_peft.md': """# Prefix-Tuning (Li & Liang, 2021)

- **Virtual Prefix Vectors**: Freezes pretrained weights and prepends continuous, learnable task-specific vectors to keys and values across all attention layers.
- **Reparameterization**: Uses temporary MLP reparameterization during training for gradient stability.
- **Parameter Efficiency**: Tunes only ~0.1% parameters while achieving full fine-tuning performance.""",

'lora_qlora_peft.md': """# LoRA & QLoRA Matrix Decomposition

- **LoRA**: Freezes base weights $W_0$ and updates weights via low-rank decomposition $\\Delta W = \\frac{\\alpha}{r} B A$ ($r \\ll d$). Merges into base weights at inference with zero added latency.
- **QLoRA**: Quantizes base weights to 4-bit NormalFloat (NF4), adds Double Quantization and Paged Optimizers, backpropagating 16-bit adapter gradients through dequantized weights.""",

'dora_weight_decomposed.md': """# DoRA: Weight-Decomposed Low-Rank Adaptation

- **Disentangling Magnitude & Direction**: Decomposes weights into magnitude $m$ and directional matrix $V$:
  $$W = m \\frac{V}{\\|V\\|_c}$$
- **Optimization**: Freezes base weights, applies LoRA to update direction $V = W_0 + BA$, and trains magnitude $m$ independently, matching full fine-tuning dynamics with zero inference overhead.""",

'model_merging_slerp_ties_dare.md': """# Model Merging: SLERP, TIES, and DARE

- **SLERP**: Spherical Linear Interpolation blends weights along high-dimensional arcs, preserving vector orientation and magnitude.
- **TIES**: Trims low-magnitude delta weights, elects sign consensus, and averages disjoint agreeing parameters.
- **DARE**: Randomly drops 90-99% of delta parameters and rescales remaining weights by $1/(1-p)$ prior to merging, removing parameter conflict.""",

'dpo_math_formulation.md': """# DPO Mathematical Formulation vs. PPO

- **DPO Formulation**: Eliminates explicit reward models by substituting the optimal policy closed-form relation into the Bradley-Terry objective:
  $$\\mathcal{L}_{\\text{DPO}} = -\\mathbb{E}\\left[\\log \\sigma\\left(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right)\\right]$$
- **Contrast with PPO**: Directly optimizes preferences via binary cross-entropy on offline pairs without actor-critic rollouts, value networks, or online RL training instabilities.""",

'kto_prospect_theory.md': """# KTO: Kahneman-Tversky Optimization

- **Prospect Theory Alignment**: Aligns models using unpaired binary feedback (desirable vs. undesirable).
- **Core Principles**: Reference dependence (evaluates utility against base reference policy), loss aversion (heavier penalties for undesirable outputs), and diminishing sensitivity (concave for gains, convex for losses).""",

'orpo_reference_free.md': """# ORPO: Reference-Free Odds Ratio Preference Optimization

- **Unified Objective**: Combines SFT loss with an odds ratio penalty comparing generation odds of winning vs. losing responses in a single training step.
- **No Reference Policy**: Eliminates the frozen reference model required by DPO, significantly reducing VRAM footprint and compute costs.""",

'simpo_length_normalized.md': """# SimPO: Length-Normalized Margin Preference Alignment

- **Length-Normalized Reward**: Sequence reward is defined as average per-token log-likelihood $r(x, y) = \\frac{\\beta}{|y|} \\log \\pi_\\theta(y|x)$, eliminating verbosity bias.
- **Target Margin**: Enforces positive target margin $\\gamma$ in Bradley-Terry loss, preventing marginal separations without reference model overhead.""",

'grpo_mechanics.md': """# GRPO: Group Relative Policy Optimization (DeepSeek-R1)

- **Critic-Free Advantage**: Samples group of outputs $\{o_1, \\dots, o_G\}$ per prompt. Computes advantage by normalizing rewards within the group:
  $$A_i = \\frac{r_i - \\text{mean}(\\{r\\})}{\\text{std}(\\{r\\})}$$
- **Efficiency**: Eliminates the separate critic/value network, halving RL memory and compute overhead while eliciting self-reflection and reasoning in DeepSeek-R1.""",

'orm_vs_prm_supervision.md': """# Outcome (ORM) vs. Process (PRM) Reward Models

- **Outcome Reward Models (ORMs)**: Evaluate final outputs only; simple binary verification, but process-blind and vulnerable to reward hacking from lucky guesses.
- **Process Reward Models (PRMs)**: Dense step-level feedback scoring every intermediate reasoning step; enables MCTS and beam search while preventing error cascades.""",

'math_shepherd_prm.md': """# Math-Shepherd: Automated Process Supervision

- **Automated Labeling**: Estimates intermediate step validity via Monte Carlo rollouts to final answers, verifying correctness against ground-truth outcomes.
- **Scalable Training**: Bypasses human step annotation bottlenecks, producing dense step-level supervision for training PRMs in Best-of-N reranking and step-level RL.""",

'rlvr_verifiable_rewards.md': """# RLVR: Reinforcement Learning with Verifiable Rewards

- **Deterministic Ground Truth**: Replaces learned subjective reward models with automated rule-based verifiers (unit tests, math engines, compilers, formal provers).
- **Self-Correction Emergence**: Policy optimization against objective binary signals forces models to learn authentic backtracking and multi-step verification without reward hacking.""",

'mcts_llm_reasoning.md': """# Monte Carlo Tree Search (MCTS) for LLM Reasoning

- **Search Topology**: Structures test-time reasoning into a search tree balancing exploration and exploitation (UCT).
- **Core Loop**: Selection -> Expansion (candidate reasoning branches) -> Evaluation (PRM or rollout simulation) -> Backpropagation.
- **Lookahead**: Enables systematic error backtracking and lookahead verification for complex mathematics and code generation.""",

'alphacode_2_clustering.md': """# AlphaCode 2: Filtering, Clustering & Selection Strategy

1. **Filtering**: Executes candidate solutions against public problem tests, eliminating ~95% failing candidates.
2. **Clustering**: Executes surviving candidates on synthetically generated test inputs, grouping programs with identical output behavior into clusters.
3. **Selection**: Samples candidates from largest behavioral consensus clusters and rates quality via scoring models to submit top 10 final solutions.""",

'o1_o3_prompting_rules.md': """# OpenAI o1 & o3 Prompting Rules

1. **No Manual CoT**: Avoid *"think step-by-step"*; models reason autonomously via internal hidden CoT. Manual CoT prompts degrade performance.
2. **Clean Constraints**: Plainly define goals, formatting rules, edge conditions, and acceptance criteria.
3. **Structured Delimiters**: Use Markdown or XML tags to cleanly separate inputs and schemas.
4. **Format-Focused Examples**: Provide clean input-output pairs without intermediate rationale traces.""",

'deepseek_r1_prompting.md': """# DeepSeek-R1 Prompting Guidelines

- **Zero-Shot Directness**: State problems directly; do not force artificial reasoning prompts or persona gymnastics.
- **Outcome Specification**: Define target output schemas (XML tags, JSON format) clearly.
- **Temperature Setting**: Recommended temperature 0.6 for creative/extended reasoning; lower for deterministic code/math tasks.
- **Structured Context**: Wrap reference context in delimiters (`<context>`, `<problem>`) to prevent prompt drift.""",

'react_trajectory_formatting.md': """# ReAct: Reason + Act Trajectory Formatting

- **Cycle**: Strict interleaving of **Thought -> Action -> Observation**:
  - `Thought`: Formulates subgoals, tracks state, and evaluates progress.
  - `Action`: Executable tool call syntax (e.g., `Search[query]`, `Lookup[term]`).
  - `Observation`: External execution output grounding next thought.
- **Termination**: Concludes when action `Finish[answer]` returns final output.""",

'pal_pot_program_prompting.md': """# PAL & PoT: Program-Aided Reasoning

- **Decoupled Execution**: LLM generates executable code expressing reasoning logic, delegating state tracking and arithmetic to an external runtime interpreter.
- **PAL**: Interleaves natural language explanations with programmatic commands.
- **PoT**: Frames the entire reasoning trace as an end-to-end program, eliminating arithmetic hallucinations.""",

'cove_factored_verification.md': """# Chain of Verification (CoVe) Factored Verification

- **Factored Pipeline**: Formulates verification questions for a baseline draft, then answers each verification question in isolated prompts WITHOUT the unverified draft present.
- **Mitigating Bias**: Decoupling drafts from verification queries prevents self-confirmation bias and hallucination snowballs prior to final response synthesis.""",

'self_refine_loop.md': """# Self-Refine Iterative Feedback Loop (Madaan et al. 2023)

- **Tripartite Loop**: Generate (initial candidate) -> Feedback (self-critique against criteria) -> Refine (synthesize improved candidate).
- **Zero Supervision**: Requires no external reward models or fine-tuning, operating entirely in-context across iterative turns for code optimization and math.""",

'rar_rephrase_respond.md': """# Rephrase and Respond (RaR, Deng et al. 2023)

- **Query Refinement**: Prompts the LLM to first rephrase and clarify ambiguous or underspecified user questions before generating answers.
- **One-Step vs. Two-Step**: Can execute within a single prompt or decouple rephrasing into a stronger model that feeds an optimized prompt into a secondary responder.""",

'meta_prompting_programming.md': """# Meta-Prompting & Prompt Programming (Reynolds & McDonell)

- **Priors Activation**: Treats foundation models as pretrained meta-learners containing rich concept distributions.
- **Prompt as Code**: Prompts act as high-level programs isolating and steering latent task distributions rather than few-shot demonstration training, maximizing zero-shot efficiency.""",

'directional_stimulus_prompting.md': """# Directional Stimulus Prompting (Li et al. 2023)

- **Dual-Model Scaffolding**: A small, lightweight policy model generates discrete hints or directional stimuli (key concepts, keywords) appended to prompts for a frozen large LLM.
- **RL Optimization**: Policy model is trained via reinforcement learning using downstream LLM rewards, steering generation without large-model parameter updates.""",

'active_prompting_uncertainty.md': """# Active Prompting (Diao et al. 2023)

- **Uncertainty Sampling**: Samples multiple reasoning paths per question to quantify predictive entropy and disagreement.
- **Selective Annotation**: Selects questions with the highest uncertainty for human CoT annotation, maximizing exemplar quality and downstream few-shot accuracy."""
}

written = 0
for fname, content in files.items():
    path = os.path.join(base, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    written += 1

print(f'Successfully wrote {written} snippet files.')
