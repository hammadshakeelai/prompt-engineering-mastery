# Research: Prompt Sensitivity and Brittleness in LLMs

## 1. Prompt Sensitivity and Brittleness
**Prompt Sensitivity** refers to the degree to which a Large Language Model's (LLM) output changes based on minor, semantically irrelevant modifications to the input prompt (e.g., adding a space, changing punctuation, or slight rephrasing like "Explain this simply" vs. "Explain this in a simple way"). Research suggests this happens because LLMs often disperse semantically similar inputs rather than clustering them, causing large shifts in log probabilities.
**Prompt Brittleness** is the practical consequence of this sensitivity. A prompt or system is considered brittle if it easily breaks or degrades in task performance when subjected to these minor variations.

## 2. The "Prompt Lottery" Phenomenon
The "prompt lottery" describes the unpredictable performance of a model based on phrasing, forcing users or evaluators to blindly test numerous variations to find the "best-performing" prompt. In academic evaluations, a high "prompt lottery" effect is a major critique; it implies that a model's high score might be due to finding the lucky optimal prompt rather than possessing generalized reasoning capabilities.

## 3. Empirical Results: "Large Language Models Are Not Robust Multiple Choice Selectors"
This ICLR 2024 paper by researchers from Tsinghua University and Tencent highlights a specific vulnerability: modern LLMs exhibit severe "selection bias" when answering multiple-choice questions (MCQs).
*   **Token Bias:** The models are highly sensitive to the position of options and show a systematic preference for specific option IDs (like always leaning toward "Option A"), regardless of the actual answer content.
*   **Mitigation (PriDe):** The authors introduced PriDe (Prior Debiasing), an inference-time technique that estimates the model's prior token bias by permuting option contents on a small set of test samples, and then subtracts this bias from the model's answer distribution.

## 4. Calibration Techniques
Calibration aims to align a model's confidence or probability scores with its actual likelihood of being correct. Techniques fall into two categories:
*   **Prompt-Based (Optimization):** Using Few-Shot Calibration (providing balanced context-free examples to remove inherent biases), Chain-of-Thought (improving logical consistency), and Self-Calibration/Evaluation (asking the model to critique its own answers).
*   **Algorithmic (Technical):** Adjusting internal parameters or output logits. Common methods include Temperature Scaling, Platt Scaling, Contextual Calibration (adjusting based on the model's response to null inputs like "[MASK]"), and using the "Invert Softmax Trick" for verbalized probabilities.

## 5. Self-Consistency (Majority Voting)
Self-Consistency is a decoding strategy used to mitigate brittleness and improve reasoning (often paired with Chain-of-Thought).
*   **Mechanism:** Instead of using greedy decoding to generate a single answer, the model uses stochastic decoding (temperature > 0) to generate multiple diverse reasoning paths for the same prompt.
*   **Majority Voting:** The final answers from all generated paths are extracted and tallied. The most frequent answer (the majority vote) is selected.
*   **Why it works:** While models might hallucinate or make errors through various flawed logical paths, the correct answer is usually reached consistently through multiple valid paths, effectively marginalizing out the noise.
