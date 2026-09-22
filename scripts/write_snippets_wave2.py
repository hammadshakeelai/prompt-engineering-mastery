import os

base = r'c:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\raw_research\snippets'
os.makedirs(base, exist_ok=True)

files = {
'mixture_of_experts.md': """# Mixture of Experts (MoE) Architecture

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
- Structural Consistency: Uniform templates ensure stable routing paths.""",

'kv_cache_flash_attention.md': """# KV Cache and Flash Attention for LLM Efficiency

## 1. KV Cache Mechanics
Past tokens Key/Value states cached in GPU memory, eliminating redundant O(N^2) prefix recomputation.
Cache size = 2 x b x s x l x h x d, dominating VRAM in long contexts.

## 2. Cache Eviction
- StreamingLLM: Retains attention sink initial tokens + rolling local window for infinite-length generation.
- H2O (Heavy Hitter Oracle): Preserves top cumulative-attention-score tokens. ~80% memory reduction with negligible degradation.

## 3. Flash Attention 2/3
Computes attention in SRAM via tiling + online softmax, avoiding O(N^2) HBM materializations.
- FA2 (A100): 50-73% peak FLOP utilization, 2x faster than FA1.
- FA3 (H100): Async TMA, FP8, warp specialization. 1.5-2x over FA2 (~75-85% utilization).

## 4. Prompt Caching APIs
- Anthropic: Explicit cache_control:ephemeral marker. Min 1024 tokens. 90% cost reduction, 80% TTFT latency cut.
- OpenAI: Automatic caching for prefixes >= 1024 tokens. 50% discount on cached tokens.""",

'speculative_decoding.md': """# Speculative Decoding for LLM Inference Speed

## Architecture
- Drafting: Small draft mechanism proposes K candidate tokens autoregressively.
- Verification: Large target model validates all K candidates in ONE parallel forward pass.

## Acceptance Mechanics
Token x accepted with probability min(1, p(x)/q(x)) where p=target, q=draft.
First rejection triggers corrected sample from max(0, p(x)-q(x)). Output is lossless.

## Advanced Architectures
- Medusa: Multiple MLP heads atop frozen target backbone. Each head k predicts token t+k. Tree attention verifies simultaneously.
- EAGLE/EAGLE-2: Speculates at feature level (hidden states). Single transformer decoder layer predicts future features. Acceptance >80%.

## Speed Gains
- Typical 2-4x wall-clock speedup in memory-bound regimes (batch=1).
- Structured outputs: Deterministic syntax (brackets, keys) pushes acceptance near ~100%, enabling >4x speedup.""",

'prompt_chaining.md': """# Prompt Chaining and Sequential Pipeline Design

Key References: Wu et al. PromptChainer (CHI 2022); Anthropic Building Effective Agents (2024)

## Decomposition
Complex tasks split into modular sequential subtasks: ingest -> outline -> draft -> critique -> verify.
Each prompt focuses on isolated sub-problem, avoiding context dilution.

## Handoff Patterns
- Direct Output-to-Input: Output of step N directly inputs to step N+1.
- State Accumulation: Shared state dict accumulates progressive outputs.
- Typed Schema Handoff: JSON/Pydantic schemas enforce interface contracts between nodes.

## Validation Gates
- Deterministic: Regex, type validation, Pydantic schema compliance.
- Semantic: LLM-as-judge evaluators for correctness and tone.
- Routing: Failed gates trigger retry loops or fallback paths.

## Frameworks
- LangChain: LCEL (prompt | model | parser), LangGraph for cyclic stateful pipelines.
- LlamaIndex: QueryPipeline (declarative DAG), Workflow with typed @step event handlers.

## Failure Modes
- Cascading hallucinations: mitigated by early-abort circuit breakers.
- Malformed outputs: recovered via OutputFixingParser with error feedback.""",

'long_context_prompting.md': """# Long Context Prompting Techniques (2024-2025)

## 1. Positional Encoding Extrapolation
- RoPE: Rotary position embeddings via rotation matrices. NTK-aware base-frequency scaling needed for extrapolation.
- YaRN: Interpolates high-frequency RoPE features, extrapolates low-frequency. Enables 128k+ context with <0.1% pre-training data.
- ALiBi: Static linear distance penalty in attention logits (-m * |i-j|). Zero-shot extrapolation but degrades beyond 64k vs. scaled RoPE.

## 2. API Context Caching
- Reuses server-side KV attention states across calls.
- Cuts prefill latency (TTFT) by 85%, input costs by 50-90%.
- Requires byte-identical prompt prefixes. Immutable context at START, dynamic user turns at END.

## 3. Structuring 100k+ Token Prompts
- Prompt Sandwiching: Core directives first, retrieved docs in middle, restate query + output constraints at end.
- Hierarchical Delimiters: XML tags prevent attention drift and prompt injection.
- Explicit Scratchpads: Instruct model to cite doc IDs in <thinking> before generating answers.""",

'raptor.md': """# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., 2024)

Citation: arXiv:2401.18059 (ICLR 2024)

## Core Concept
Standard RAG retrieves short isolated chunks, failing on questions requiring holistic synthesis.
RAPTOR builds a multi-level tree by recursively clustering and summarizing chunks bottom-up.

## 1. Hierarchical Clustering
- Chunking & Embedding: Documents divided into leaf chunks (~100 tokens), embedded via SBERT.
- UMAP + GMM Soft Clustering: Reduces dimensionality, then GMM allows overlapping multi-cluster membership.
- Dynamic K: Bayesian Information Criterion (BIC) selects optimal cluster count.

## 2. Recursive Summarization
- LLM synthesizes each cluster's chunks into an abstractive summary node.
- Summaries re-embedded, reclustered, summarized iteratively, forming layers of increasing abstraction.

## 3. Improving Multi-Hop Reasoning
- Tree Traversal: Top-down branch pruning OR collapsed tree search (query all leaf + summary nodes via cosine similarity).
- Global + Local Context: Summary nodes provide macro-level context; leaf nodes provide granular evidence.
- Results: Up to +20% accuracy on QuALITY, significant gains on NarrativeQA and QASPER.""",

'llm_calibration.md': """# LLM Calibration and Verbalized Uncertainty

## Expected Calibration Error (ECE)
Partitions N predictions into M confidence bins and computes weighted absolute difference:
ECE = sum_m (|B_m|/N) * |acc(B_m) - conf(B_m)|
Lower ECE = better calibration (0 = perfect).

## Temperature Scaling
Post-processing calibration rescaling pre-softmax logits by learned scalar T > 0.
Optimized via NLL on validation data. T > 1 softens overconfident logits without altering argmax.

## Verbalized Uncertainty
Elicits confidence via natural language ("How confident are you on a 0-100% scale?").
- Flaws: Prone to overconfidence, mode collapse around 80%/100%, prompt fragility.
- RLHF/instruction tuning often degrades verbalized calibration.

## Kadavath et al. (2022) - Anthropic Self-Knowledge Study
Large models are well-calibrated when predicting P(True | question, answer) via logit evaluation.
Calibration improves with model scale and updates with external context.

## Techniques for Well-Calibrated Estimates
1. P(True) Logit Probing: Query token probability instead of generated text.
2. Post-Hoc Recalibration: Temperature scaling, Platt scaling, isotonic regression.
3. Semantic Entropy: Cluster multiple completions by semantic meaning, measure consistency.
4. Calibrated Prompting: Few-shot exemplars with diverse calibrated probabilities + reasoning before confidence.""",

'prompt_injection.md': """# Prompt Injection and Indirect Prompt Injection Attacks

## 1. Mechanical Attack Vector
Exploits conflation of control plane (system instructions) and data plane (user/retrieved text).
Both serialized into one autoregressive sequence; transformer cannot distinguish meta-instructions from passive data.
Adversarial tokens mimicking authoritative syntax hijack model's objective function at runtime.

## 2. Greshake et al. (2023) Findings
Paper: "Not what you've signed up for" (arXiv:2302.12173)
Formalized Indirect Prompt Injection (IPI): attackers embed payloads in external resources (webpages, emails, APIs) retrieved by the agent.
Demonstrated: silent data exfiltration via Markdown image rendering, arbitrary tool execution, session contamination, self-replicating prompts (LLM worms).

## 3. Indirect Injection via Web Content
When agents scrape web content, hidden DOM elements (display:none) or HTML comments embed instructions.
Upon ingestion, payload overrides initial system goal, coercing model to leak PII or invoke destructive tools.

## 4. Defensive Patterns
- Input Sanitization: XML/JSON structural delimitation, nonces/spotlighting, perimeter classifiers (SecAlign, Llama Guard).
- Privilege Separation (Dual-LLM): Quarantined LLM parses untrusted data, returns non-executable content; separate Privileged LLM retains tool/API access.""",

'llm_watermarking.md': """# LLM Output Watermarking

## Red-Green Token Watermarking (Kirchenbauer et al., 2023)
Partitions vocabulary into green list G and red list R via pseudo-random hash of preceding tokens.
Additive bias delta > 0 injected into green token logits before softmax, biasing sampling toward G.

## Semantic Watermarking
Hash-based token watermarks break under lexical edits.
SemStamp ties partitioning to semantic embeddings / sentence-level latent spaces, maintaining robustness across paraphrase boundaries.

## Detection: Z-Score Statistical Test
z = (|s|_G - gamma*T) / sqrt(T*gamma*(1-gamma))
where |s|_G = observed green tokens across length T.
Standard threshold z >= 4 (p < 3.2e-5) provides bounded false positive rates.

## Limitations
- Low-Entropy: Deterministic outputs (code, math) suffer quality loss or insufficient green tokens.
- Evasion: Vulnerable to heavy paraphrasing, translation round-trips, token insertion/deletion.
- Length: Short texts (T < 50 tokens) lack statistical power.

## Implications for Attribution
Enables zero-bit / multi-bit provenance tracking for copyright, disinformation, academic integrity.
Open-weights models can bypass; key compromise enables spoofing.""",

'sycophancy.md': """# Sycophancy in LLMs and Mitigation

## Definition
Tendency to tailor outputs to flatter/confirm user opinions even when user is factually incorrect.

## Perez et al. (2022) Findings (Anthropic)
- Models sacrifice factual accuracy to mimic user's expressed viewpoint.
- Sycophancy scales with model size: larger models exhibit stronger sycophancy.
- Manifests in both pre-trained base models and instruction-tuned models.

## RLHF Amplification
Human evaluators prefer polite, agreeable responses over critical corrections.
Reward models penalize disagreement, teaching the policy that validating users maximizes reward.

## Constitutional AI Mitigation
CAI replaces human rater bias with model self-critique guided by explicit principles:
truthfulness, objective neutrality, non-evasiveness over user appeasement.
RLAIF reward model evaluates against constitutional tenets rather than human approval.

## Prompting Strategies
1. Critical/Adversarial Persona: "Point out flaws and false assumptions in my argument."
2. Neutral/Third-Party Framing: Frame queries objectively or as a third party evaluation.
3. Structured Analysis: Require explicit pros/cons before conclusion.
4. Explicit Directives: "Prioritize truth over politeness; correct errors directly." """,

'multi_agent_debate.md': """# Multi-Agent Debate and Society of Mind Prompting

Citations: Du et al. (2023, arXiv:2305.14325); Chan et al. ChatEval (2023, arXiv:2308.07201)

## Du et al. (2023) Framework
Inspired by Marvin Minsky's Society of Mind. Multiple independent LLM instances:
1. Independently generate initial reasoning trajectories and solutions.
2. Exchange and critique each other's outputs over multiple rounds.
3. Iteratively incorporate peer arguments to correct logical fallacies and reach consensus.

## Key Architectures
- ChatEval: Multi-agent referee team with diverse personas to evaluate text quality. One-on-one or simultaneous-talk topologies.
- Ensemble Refinement: Multiple reasoning paths pooled as student reasoning; LLM iteratively refines and synthesizes a final response.

## When Debate Helps
- Complex multi-step reasoning (math, code, logic) where intermediate validation is required.
- Mitigating hallucinations through cross-examination and diverse perspectives.

## When Debate Hurts
- Agreement Bias: Models converge prematurely on plausible but incorrect peer solutions.
- Debate Hacking: Agents optimize for rhetoric over truthfulness; overconfident agents dominate.
- Efficiency: Exponential token/latency overhead often fails to outperform simple self-consistency.""",

'universal_self_consistency.md': """# Universal Self-Consistency (Chen et al., 2023)

Citation: arXiv:2311.17311

## Core Concept
Standard Self-Consistency uses exact-match majority voting, restricting to closed-domain discrete answers.
Universal Self-Consistency (USC) extends to arbitrary free-form text by replacing parsing with model-based semantic consensus.

## Mechanism: LLM as Selector
1. Diverse Sampling: Generate N candidate responses using temperature sampling + CoT.
2. Contextual Aggregation: Concatenate original query + all candidate responses into a single prompt.
3. Consensus Selection: LLM selector evaluates semantic consistency, mutual agreement, reasoning validity across the pool.

Evaluating mutual consistency is cognitively easier for LLMs than verifying absolute correctness in isolation.

## Advantages
- Free-Form Text: Works on summarization, open-ended QA, creative reasoning without discrete answer extraction.
- No Parsers: Eliminates fragile regex, keyword extractors, and external execution verifiers.
- Breadth: Matches standard majority voting on math, competitive with execution-based voting on code.""",

'chain_of_verification.md': """# Chain-of-Verification (CoVe) - Dhuliawala et al. (2023)

Citation: arXiv:2309.11495 (Meta AI)

## Four-Step Pipeline
1. Generate Baseline Response: Initial draft answer to the query.
2. Plan Verifications: Formulate targeted verification questions to fact-check claims in the draft.
3. Execute Verifications: Answer each verification question independently.
4. Generate Final Response: Revised output incorporating verified facts and correcting hallucinations.

## Joint vs. Factored Variants
- Joint Verification: Planning and answering combined in one prompt WITH the baseline draft present.
  Limitation: Self-confirmation bias causes model to repeat hallucinated errors.
- Factored Verification: Each question answered as an independent prompt ISOLATED from the baseline draft.
  Advantage: Decoupling prevents attending to own hallucinations, dramatically increasing accuracy.
- Factor + Revise: Factored answering + explicit cross-checking step before final response.

## Results
- Reduces hallucinations on Wikidata list generation, MultiSpanQA, and biography generation.
- Factored consistently outperforms joint verification.
- Training-free, test-time only.""",

'test_time_compute_ext.md': """# Test-Time Compute Scaling and Reasoning Models

## 1. Test-Time Compute Scaling
Trades inference compute for accuracy by generating dynamic hidden Chain-of-Thought before final answers.
Models self-correct, explore hypotheses, and backtrack.
Scaled along two axes:
- Sequential: Longer CoT chains / thinking budgets.
- Parallel: Best-of-N, tree search / MCTS.
Performance scales log-linearly with inference compute on complex benchmarks.

## 2. Reinforcement Learning with Verifiable Rewards (RLVR)
Trains reasoning models using deterministic ground-truth verifiers (compilers, unit tests, symbolic math engines).
No subjective human preferences, no reward hacking, no labeling bottlenecks.
Policy autonomously discovers search strategies and extended reasoning trajectories.

## 3. DeepSeek-R1: Outcome-Based RL
Abandoned neural Process Reward Models (PRMs) due to:
- Cannot consistently define step boundaries for general tasks.
- High annotation cost and reward hacking vulnerability.
Uses rule-based outcome verification (accuracy + formatting) via Group Relative Policy Optimization (GRPO).
Proved pure outcome-based RL naturally elicits self-reflection and verification without step-level supervision.

## 4. Prompting Thinking Models
- Avoid "Think Step-by-Step": Models natively deliberate; manual CoT triggers degrade quality.
- Focus on Constraints and Intent: State objectives, format specs, and edge conditions plainly.
- Use Delimiters: XML tags or Markdown headers structure inputs and background documents.
- Provide Context, Not Solution Paths: Supply complete reference info; leave reasoning to the model.""",

'decomposed_prompting.md': """# Decomposed Prompting (DecomP) - Khot et al. (ICLR 2023)

Citation: arXiv:2210.02406

## Core Concept
Solves complex reasoning by decomposing into structured sub-tasks orchestrated by a central decomposer.
Generates a program-like execution trace where sub-task inputs/outputs are passed as variables.

## Sub-Task Handlers
- Specialized handlers: distinct few-shot LLM prompts, fine-tuned models, or symbolic tools/APIs.
- Modularity: Handlers independently optimized, debugged, or replaced without altering the main architecture.

## Recursive Decomposition
- When a handler encounters inputs exceeding capability, it recursively invokes the decomposer.
- Base case reduction: Hierarchical sub-problems unfold until manageable.

## DecomP vs. Least-to-Most Prompting
1. Architecture: Least-to-Most uses ONE LLM throughout. DecomP routes sub-tasks to HETEROGENEOUS handlers.
2. Control Flow: Least-to-Most is strictly flat and linear. DecomP supports non-linear control flow, variable binding, recursive trees.""",

'analogical_prompting.md': """# Analogical Prompting (Yasunaga et al., ICLR 2024)

Citation: arXiv:2310.01714 (Stanford / Google DeepMind)

## Connection to Human Analogical Reasoning
Humans resolve novel problems by retrieving structurally analogous past experiences and transferring reasoning strategies.
Analogical prompting mimics this by guiding LLMs to recall relevant problem schemas from parametric memory.

## Self-Generated Analogous Problems
Traditional few-shot CoT relies on static human-labeled exemplars.
Analogical prompting directs the LLM to autonomously recall/generate 3-5 relevant exemplars tailored on-the-fly:
- Analogous problems
- Step-by-step reasoning rationales
- Solutions matching the input problem's sub-domain, difficulty, and algorithmic patterns

## Outperforming Few-Shot CoT
- Consistently outperforms 0-shot CoT and matches/surpasses manual few-shot CoT.
- Benchmarks: GSM8K, MATH, Codeforces competitive programming.
- Dynamic Context Adaptation: Self-generated exemplars cover long-tail theorems and data structures.
- Zero Labeling Overhead: Eliminates need for curated few-shot exemplar repositories.""",

'attention_mechanisms.md': """# Attention Mechanisms and Positional Encodings for Prompting

## 1. Dot-Product Attention and Token Saliency
A = softmax(QK^T / sqrt(d_k)) * V
Query-key dot products measure semantic compatibility. Softmax exponentially amplifies top-scoring pairs.
Tokens whose keys align strongly with generation queries capture attention mass; noisy tokens suffer dilution.

## 2. Absolute vs. Relative Encodings
- Absolute (Sinusoidal/Learned): Fixed index coordinates, fail to extrapolate beyond pre-trained context window.
- Relative (ALiBi, T5 bias): Inject distance offsets directly into attention logits, generalizing robustly.
- RoPE: Rotates Q and K in complex 2D sub-spaces. Preserves vector norms, naturally encodes relative distance.
  Permits context window expansion via frequency scaling (YaRN, NTK).

## 3. Practical Prompt Structure Implications
- Boundary Placement (Sandwiching): Anchor core system rules at token 0; place queries/schema at final tokens.
- Delimiters as Attention Anchors: XML tags (<context>, <doc>) create distinct key vectors query heads latch onto.
- Salience vs. Distractors: Softmax is zero-sum; extraneous tokens siphon attention away from critical instructions.
- Attention Sinks: Token 0 absorbs excess attention mass in many models (StreamingLLM finding).""",

'buffer_of_thoughts.md': """# Buffer of Thoughts (BoT) - Yang et al. (NeurIPS 2024)

Citation: arXiv:2406.04271

## Core Motivation
ToT achieves high accuracy but requires 100+ queries.
Standard CoT generates thoughts from scratch without cross-task memory.
BoT introduces thought-augmented reasoning with a persistent meta-buffer.

## Architecture
1. Problem Distiller: Extracts critical task specifications, constraints, and mathematical relationships.
2. Meta-Buffer (Thought-Template Library): Stores high-level thought-templates - distilled meta-thoughts and algorithmic patterns (dynamic programming, divide-and-conquer, proof-by-contradiction) abstracted from previous solutions.
3. Buffer-Manager: Maintains, refines, and deduplicates templates. Prevents memory drift.
4. Adaptive Instantiation: Retrieves most relevant template and instantiates into concrete multi-step reasoning path.

## Efficiency and Performance
- Consumes ~12% of the compute cost of Tree of Thoughts.
- Outperforms CoT and matches/surpasses ToT on Game of 24, 3D spatial reasoning, Checkmate-in-One.
- Enables smaller models (Llama-3-8B) to outperform larger models (Llama-3-70B) via structured template guidance.

## Vs. Step-Back Prompting
Step-Back abstracts on-the-fly. BoT catalogs persistent distilled abstractions for cross-task retrieval.""",
}

written = 0
for fname, content in files.items():
    path = os.path.join(base, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    written += 1

print(f'Successfully wrote {written} snippet files.')
