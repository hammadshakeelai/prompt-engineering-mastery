# Chain-of-Verification (CoVe) - Dhuliawala et al. (2023)

Citation: arXiv:2309.11495 (Meta AI)

## Four-Step Pipeline
1. Generate Baseline Response: Initial draft answer to the query.
2. Plan Verifications: Formulate targeted verification questions to fact-check claims in the draft.
3. Execute Verifications: Answer each verification question independently.
4. Generate Final Response: Revised output incorporating verified facts and correcting hallucinations.

## Joint vs. Factored Variants
- Joint Verification: Planning and answering combined in one prompt WITH the baseline draft present.
  Limitation: Self-confirmation bias causes model to repeat hallucinated errors.
- Factored Verification: Each question answered as an independent prompt ISOLATED from the baseline draft.
  Advantage: Decoupling prevents attending to own hallucinations, dramatically increasing accuracy.
- Factor + Revise: Factored answering + explicit cross-checking step before final response.

## Results
- Reduces hallucinations on Wikidata list generation, MultiSpanQA, and biography generation.
- Factored consistently outperforms joint verification.
- Training-free, test-time only.