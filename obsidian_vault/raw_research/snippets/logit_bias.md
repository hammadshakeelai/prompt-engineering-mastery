# Logit Bias in Constrained Generation

## Mechanism
Logit bias modifies token generation probabilities at inference time prior to softmax. For logit vector z, an additive scalar bias b_i is injected for target token i:
P(w_i) = exp(z_i + b_i) / sum_j exp(z_j + b_j)

- **Suppression (b_i << 0):** Setting -100 drives probability to 0, banning the token.
- **Forcing (b_i >> 0):** Setting +100 virtually guarantees token emission.
- **Calibration:** Moderate values [-5, +5] guide style without strict forcing.

## Use Cases
1. **Categorical Constraints:** Restrict output to "Yes"/"No" or MCQ options A/B/C/D.
2. **Grammar Enforcement:** Outlines/Guidance apply dynamic state-dependent logit masks at each step.
3. **Safety Guardrails:** Ban toxic strings, PII, or premature refusal prefixes.
4. **Flow Control:** Force/suppress special tokens (tool-call delimiters, EOS markers).