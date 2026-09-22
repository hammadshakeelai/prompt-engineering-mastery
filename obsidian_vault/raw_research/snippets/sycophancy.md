# Sycophancy in LLMs and Mitigation

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
4. Explicit Directives: "Prioritize truth over politeness; correct errors directly." 