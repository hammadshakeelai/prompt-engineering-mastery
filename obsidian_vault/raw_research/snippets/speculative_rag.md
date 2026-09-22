# Speculative RAG (Draft-Verify Pipelines)

Accelerates and improves RAG using a draft-and-verify paradigm:
- Retrieved documents are partitioned into distinct subsets.
- A smaller, faster "drafter" processes subsets in parallel to generate multiple diverse candidate answers.
- A larger "verifier" evaluates and validates all candidate drafts in a single pass alongside query and rationale, selecting or synthesizing the final answer.
Reduces inference latency, cuts compute costs, and mitigates distractor noise.