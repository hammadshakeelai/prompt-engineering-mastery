# Test-Time Compute Budget Forcing & Optimal Verifier Allocation (Snell et al., 2024)

## 1. Inference FLOPs vs Pretraining Parameter Scaling
Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar (*Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*, UC Berkeley & Google DeepMind, 2024 / arXiv:2408.03314) formalize inference compute as an independent scaling axis:
$$\text{FLOPs}_{\text{total}} = \text{FLOPs}_{\text{pretrain}} + Q \times \text{FLOPs}_{\text{test-time}}$$
In complex reasoning domains, scaling test-time compute on smaller models matches or surpasses large models evaluated single-pass.

## 2. Two Orthogonal Test-Time Mechanics
1. **Verifier-Guided Search (Parallel Best-of-$N$ & PRM Beam Search):**
   - Samples $N$ trajectories and scores them via Process Reward Models (PRMs).
   - Probability of finding at least one valid trajectory: $P_{\text{success}} = 1 - (1 - p)^N$.
   - *Limitation:* When base pass@1 $p \to 0$, Best-of-$N$ requires exponential draws and succumbs to verifier false-positive exploitation (Goodharting).
2. **Adaptive Sequence Revision (Sequential Correction):**
   - Conditioned on prior attempt and critique: $y^{(k+1)} \sim \pi_{\text{revise}}(y \mid x, y^{(k)}, c^{(k)})$.
   - Performs surgical edits on flawed reasoning steps, maintaining valid partial derivations.

## 3. Prompt Difficulty-Conditioned Optimal Allocation
Snell et al. demonstrate that search optimality depends strictly on prompt difficulty $d(x)$:
- **Easy Problems ($p \ge 0.5$):** Parallel Best-of-$N$ with low $N$ is optimal; revision adds unnecessary compute overhead.
- **Intermediate Problems ($0.15 \le p < 0.5$):** Verifier-guided step-level beam search prunes branches before errors compound.
- **Hard / Out-of-Distribution ($p < 0.05$):** Sequential revisions are required to break outside the base policy's narrow zero-shot distribution.

## 4. Empirical Benchmarks & Parameter Equivalence
- **Compute Efficiency:** Difficulty-adaptive compute routing achieves target MATH / GSM8K accuracy with **$>4\times$ less compute** than uniform Best-of-$N$.
- **Parameter Invariance:** A **7B base model** with compute-optimal inference search matches or outperforms a **$14\times$ larger model (70B+)** evaluated greedily with equivalent total compute.
