# LLM Calibration and Verbalized Uncertainty

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
4. Calibrated Prompting: Few-shot exemplars with diverse calibrated probabilities + reasoning before confidence.