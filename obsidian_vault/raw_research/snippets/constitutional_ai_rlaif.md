# Constitutional AI & RLAIF (Bai et al. 2022)

## 1. Motivation: Eliminating Human Feedback Bottlenecks
Standard RLHF relies on costly, non-scalable, and inconsistent human preference annotations. **Constitutional AI (CAI)** replaces human annotators with a set of explicit, written normative principles (the "Constitution") to steer models toward helpfulness and harmlessness autonomously.

## 2. Two-Stage Training Topology
1. **Supervised Learning (SL) Phase (Critique & Revision):**
   - The base model generates an initial candidate response $y_0 \sim p(y \mid x)$ to a red-team prompt $x$.
   - The model is instructed to critique its response against a specific constitutional principle $C_k$:
     $$c \sim p(c \mid x, y_0, C_k)$$
   - The model generates a revised output:
     $$y_{\text{revised}} \sim p(y \mid x, y_0, c, C_k)$$
   - The base model is supervised fine-tuned (SFT) on pairs $(x, y_{\text{revised}})$, internalizing harmless behaviors without human feedback.

2. **Reinforcement Learning from AI Feedback (RLAIF) Phase:**
   - The SFT model generates paired candidate responses $(y_1, y_2)$ for prompt $x$.
   - A feedback model scores both completions according to constitutional criteria, producing preference probabilities $P_{\text{AI}}(y_1 \succ y_2 \mid x, C)$.
   - A Preference Model (PM) is trained via Bradley-Terry loss:
     $$\mathcal{L}_{\text{PM}} = -\mathbb{E}\left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right]$$
   - Policy $\pi_\theta$ is optimized via PPO or DPO against this synthetic reward model with KL regularization $\beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$.

## 3. Empirical Results & Scaling
- Achieved Pareto improvements on Anthropic's harmlessness evaluations without degrading helpfulness.
- Eliminates annotator exposure to toxic content while enabling transparent, auditable modification of safety guardrails by simply editing text constitutional rules.
