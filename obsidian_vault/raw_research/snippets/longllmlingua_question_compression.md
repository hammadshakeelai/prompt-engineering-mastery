# LongLLMLingua: Question-Aware Prompt Compression (Jiang et al., Microsoft / ACL 2024)

## 1. Positional Degradation & Context Dilution
In long-context RAG, LLMs suffer from the "Lost in the Middle" attention degradation (Liu et al., 2023), forgetting details located in intermediate context positions. Unconditional compression prunes tokens that appear statistically predictable in isolation but are crucial for answering specific query $q$.

## 2. Question-Conditioned Mutual Information
LongLLMLingua (Jiang et al., ACL 2024 / arXiv:2310.06839) scores token retention via contrastive conditional perplexity:
$$I(x_i; q) = \log \frac{P_{\mathcal{M}}(x_i \mid x_{<i}, q)}{P_{\mathcal{M}}(x_i \mid x_{<i})}$$
Tokens whose predictability sharply increases given $q$ are preserved. Dynamic budgets allocate higher retention rates to documents with high aggregate query mutual information.

## 3. Boundary Reordering Protocol
Reorders compressed passages so that the top-relevance document is positioned at the prompt suffix adjacent to the question, the second-ranked document is placed at the prefix start, and lower-scoring documents reside in the interior.

## 4. Empirical Performance
- Delivers a **$+21.4\%$ accuracy boost** on NaturalQuestions with **$4\times$ fewer tokens** on GPT-3.5-Turbo.
- Slashes API financial cost by up to **$94\%$** on LooGLE while accelerating end-to-end latency by $1.4\times\text{--}2.6\times$.
