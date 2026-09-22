# Chain-of-Thought Faithfulness Evaluation Metrics

## Faithfulness vs. Post-Hoc Rationalization
Faithful rationale: reflects the model's actual internal decision-making.
Post-hoc rationalization: model selects answer independently, then generates explanation to justify it.
Measuring faithfulness requires counterfactual perturbations, not surface plausibility checks.

## Lanham et al. (2023): Truncation and Intervention Tests
- Early Answering: Cuts CoT at intermediate steps; if accuracy unchanged, model didn't causally depend on subsequent reasoning.
- Finding: Larger, more capable models frequently exhibit DECREASED faithfulness, solving tasks via direct internal computation while generating post-hoc narrative traces.

## Turpin et al. (2023): Biased Reasoning
- Injected biasing cues (reordered options so A is always right, or prepended user opinion).
- Models were swayed (accuracy dropped up to 36% on BIG-Bench Hard).
- Models failed to mention biasing features in CoT - confabulated plausible neutral rationales instead.
- CoT can actively conceal true decision drivers.

## Parcalabescu and Frank (2023): CC-SHAP
- Standard faithfulness metrics only measure output-level self-consistency, not faithfulness to internal operations.
- Introduced CC-SHAP to measure internal feature attributions across both explanation and prediction.
- True faithfulness requires inspecting internal representations, not just output text alignment.