---
name: mixture-of-agents-architect
description: Specialized directive for designing, deploying, and optimizing Mixture-of-Agents (MoA) layered collaborative topologies, proposer-aggregator coordination, heterogeneous model routing, and collective intelligence consensus.
---

# Mixture-of-Agents Architect Skill

Use this skill when designing multi-model collective intelligence systems, orchestrating heterogeneous LLM ensembles, implementing layered feedforward multi-agent pipelines (MoA), managing context aggregation across proposers, or configuring master aggregators to maximize factual consensus and benchmark performance.

## 1. Core Principles of Mixture-of-Agents (MoA)

1. **The LLM Collaborativeness Phenomenon:**
   - LLMs generate significantly higher quality responses when conditioned on candidate generations from other models, even when those candidate models exhibit lower individual benchmark scores than the evaluator.
   - Cross-model exposure breaks individual model reasoning blind spots and exposes orthogonal perspectives.

2. **Heterogeneity over Homogeneity:**
   - Never ensemble identical models or simple temperature variations if heterogeneous architectures are available.
   - Maximize diversity by mixing distinct model families (e.g., Qwen, Llama, Mistral/Mixtral, DeepSeek). Distinct pretraining data distributions provide uncorrelated error residuals $\operatorname{Cov}(\epsilon_i, \epsilon_j) \approx 0$, allowing factual truth $\mu(x)$ to dominate.

3. **Separation of Exploration & Exploitation:**
   - **Proposer Layers:** Prioritize high entropy, structural diversity, and domain-specific knowledge. Smaller, cost-efficient models (e.g., 7B–14B or MoE variants) can act as highly effective proposers.
   - **Aggregator Layer:** Prioritize strong instruction following, logical consistency, and critical evaluative attention (e.g., 70B+ class models) to reconcile contradictions and eliminate hallucinations.

## 2. Layered Feedforward Architecture & Scheduling

An MoA topology is structured as an $L$-layer Directed Acyclic Graph (DAG) with $M_l$ proposers per layer:

```
[User Query x]
      │
      ├───► [Proposer A_(1,1)] ───► Candidate y_(1,1) ───┐
      ├───► [Proposer A_(1,2)] ───► Candidate y_(1,2) ───┤
      └───► [Proposer A_(1,M)] ───► Candidate y_(1,M) ───┤
                                                         │
                                               [Context Aggregation Y_1]
                                                         │
      ┌──────────────────────────────────────────────────┴──────────────────┐
      ▼                                                                     ▼
[Refiner A_(2,1)(x, Y_1)]                                         [Refiner A_(2,2)(x, Y_1)]
      │                                                                     │
      └──────────────────────────────────┬──────────────────────────────────┘
                                         ▼
                             [Context Aggregation Y_2]
                                         │
                                         ▼
                          [Master Aggregator A_agg(x, Y_2)]
                                         │
                                         ▼
                             [Terminal Consensus y*]
```

### Context Aggregation Prompting Protocol
When formatting inputs for intermediate proposers and the master aggregator, use the following standardized XML template:

```markdown
You have been provided with candidate answers from multiple independent models to the user's question below.
Your task is to synthesize, critique, and produce an answer that is strictly superior to all individual candidates.

[User Query]
{query}

[Candidate Responses]
Candidate 1:
{candidate_1}

Candidate 2:
{candidate_2}

...

[Synthesis Instructions]
1. Identify factual consensus and cross-verify claims across all candidates.
2. If candidates conflict on mathematical derivations or facts, independently derive the solution to identify the error.
3. Eliminate hallucinations, omissions, and redundant filler.
4. Output only the finalized, comprehensive, and well-structured answer.
```

## 3. Production Inference & KV Cache Optimization

1. **Prefix KV Cache Sharing:**
   - In Layer 1, all $M_1$ agents process identical prompt $x$. Leverage SGLang RadixAttention or vLLM automatic prefix caching to share prefill KV caches across concurrent requests.
2. **Layer Budget & Sizing Directives:**
   - Optimal default configuration: $L=2\text{--}3$ layers, $M=3\text{--}4$ proposers per layer.
   - Exceeding $M=6$ proposers yields diminishing returns while bloating the aggregator's context window and triggering "lost in the middle" attention degradation.
3. **Candidate Pruning & De-duplication:**
   - Compute fast semantic embeddings across candidates. If two candidates exhibit cosine similarity $>0.92$, drop the shorter one before assembling the context pool for Layer 2.

## 4. Verification & Benchmarking Checklist

- Evaluate final outputs against length-controlled benchmarks (**AlpacaEval 2.0 LC Win Rate** and **Arena-Hard-Auto**) to ensure multi-agent consensus reflects genuine reasoning rather than verbosity expansion.
- Verify that aggregator models actively filter, rather than concatenate, conflicting candidate claims.
