# Semantic Entropy & Epistemic Uncertainty (Farquhar et al., Nature 2024)

## 1. The Confound of Lexical Variation
Evaluating model confidence using token-level Shannon entropy $H(S \mid x) = -\sum P(s \mid x) \ln P(s \mid x)$ conflates syntactic, stylistic, and formatting variation with factual uncertainty. A model with high lexical variety can be completely certain of the underlying semantic fact, while a model with low token entropy may produce memorized hallucinations.

## 2. Semantic Equivalence Partitioning & Formulation
Lorenz Kuhn et al. (ICLR 2023) and Sebastian Farquhar et al. (*Detecting Hallucinations in Large Language Models Using Semantic Entropy*, Nature 2024) quantify uncertainty over **meaning equivalence classes**:

1. **Bidirectional Entailment Partitioning:**
   Sample $N$ completions $s^{(1)}, \dots, s^{(N)} \sim P(s \mid x)$. Partition them into disjoint clusters $C_1, \dots, C_K$ where:
   $$s^{(i)} \sim s^{(j)} \iff \text{Entails}(s^{(i)}, s^{(j)}) \land \text{Entails}(s^{(j)}, s^{(i)})$$
   evaluated via a bidirectional Natural Language Inference (NLI) model.

2. **Cluster Probability Marginalization:**
   $$P(C_k \mid x) = \sum_{s \in C_k} P(s \mid x) \approx \frac{1}{N} \sum_{i=1}^N \mathbb{I}(s^{(i)} \in C_k)$$

3. **Semantic Entropy ($\text{SE}$):**
   $$\text{SE}(x) = -\sum_{k=1}^K P(C_k \mid x) \ln P(C_k \mid x)$$

## 3. Key Properties & Empirical Superiority
- **Invariance to Paraphrase:** If all generations share the identical semantic truth despite differing vocabulary, $K = 1$ and $\text{SE}(x) = 0$.
- **Confabulation Detection:** Achieves **$0.85\text{--}0.92$ AUROC** on TriviaQA, CoQA, and BioASQ, outperforming token entropy, perplexity, and prompt self-reflection by **$10\text{--}20$ AUROC points**.
- **Task-Agnostic Guardrail:** Operates zero-shot without fine-tuning, providing a mathematically grounded filter for hallucination in high-stakes reasoning agents.
