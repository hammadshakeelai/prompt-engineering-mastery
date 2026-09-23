# DeepSeek-R1-Zero: Pure RL Reasoning Emergence (Guo et al., 2025)

## 1. Zero-SFT Reinforcement Learning
DeepSeek-R1-Zero (arXiv:2501.12948) demonstrates that sophisticated mathematical and algorithmic reasoning can emerge spontaneously from a raw base language model (DeepSeek-V3-Base) via large-scale reinforcement learning without preliminary Supervised Fine-Tuning (SFT).

## 2. Algorithmic Formulation & Reward Architecture
1. **Group Relative Policy Optimization (GRPO):**
   Samples a group of $G$ outputs $\mathcal{O} = \{o_1, \dots, o_G\}$ per prompt $q$. Computes advantage via group statistics:
   $$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\}) + \epsilon}$$
   eliminating the memory footprint of an auxiliary critic network.

2. **Rule-Based Reward Functions:**
   - **Accuracy Reward ($r_{\text{acc}}$):** Evaluates mathematical answers via CAS compilers (SymPy) or code via isolated unit test execution.
   - **Format Reward ($r_{\text{format}}$):** Enforces `<think> ... </think>` structure.
   - Zero reliance on neural reward models (PRMs/ORMs) or human preference labels.

## 3. The "Aha Moment" & Cognitive Phase Transitions
- **Autonomous Compute Scaling:** Model reasoning length autonomously scales from $\sim 700$ tokens to over $10,000$ tokens without explicit length bonuses.
- **Dynamic Self-Correction:** Spontaneously discovers backtracking (`"Wait, let me double check that..."`), lemma verification, and alternative search path exploration.
- **Dense Distillation:** Reasoning traces distilled into standard dense models (Qwen-2.5-32B) score **$72.6\%$ on AIME 2024 and $94.3\%$ on MATH-500**, matching proprietary frontier systems.
