---
namespace: bt6-maintainer
name: bt6-release-readiness
platforms: [all]
description: Harden an untagged BT6 release candidate to the exhaustive release bar after pull requests were accepted through the bounded contributor gate.
triggers:
  - prepare a BT6 release
  - harden a BT6 release candidate
  - audit merged PRs for release
  - improve coverage before release
requires:
  - candidate-reference: an exact candidate commit and comparison base
  - repository-context: canonical repository, CI remote, and validation profile resolvable from project state
ensures:
  - release-gaps-audited: merged behavior and risk surfaces are checked against the full release policy
  - maintainer-gaps-repaired: authorized test and correctness gaps are fixed before tagging
  - candidate-not-certified: readiness evidence never substitutes for exact-tag validation
---

# BT6 Release Readiness

Prepare a release candidate after normal pull requests have passed the bounded
`validation.quick` contributor gate. Apply `bt6-maintainer-guardrails`. This
workflow may improve code, tests, documentation, and repository-owned release
policy when the operator authorizes implementation. It does not create or
promote a tag.

## Required context

1. Resolve the canonical repository, base branch, CI remote, profile, candidate
   commit, comparison base, and current authorization.
2. Require a clean or explicitly understood working tree. Preserve unrelated
   work and use an isolated checkout when validation could overwrite it.
3. Inventory merged changes since the previous release or configured comparison
   base. Map production changes to behavior tests, ownership, and risk surfaces.
4. Treat PR acceptance as evidence of reviewability only. Do not infer release
   readiness from quick checks, aggregate coverage, or merge status.

## Hardening loop

1. Run `validation.full` plus applicable documentation, research-integrity,
   risk-surface, packaging, compatibility, supply-chain, repeat, mutation, and
   conditional-platform checks that can run before tagging.
2. Measure repository line and branch coverage, configured mature/critical
   scope, changed or touched-module regression, and new-module floors. Record
   skipped, unavailable, or stale evidence explicitly.
3. Prioritize failures and coverage gaps by release risk, recent change, public
   contract, trust boundary, and defect history—not by raw uncovered-line count.
4. For each blocking gap, add outcome-oriented tests and correct the underlying
   implementation or policy defect. Never weaken a release threshold to make a
   candidate pass.
5. Re-run focused checks after each repair, then re-run the complete applicable
   pre-tag gate. Stop when all locally executable hard gates pass or a required
   external/platform gate remains unresolved.
6. When `releaseEvidence` is configured, require one deterministic source ZIP
   of the tested commit, a canonical SHA-256 manifest, a bound source SBOM,
   SLSA/in-toto provenance, and the configured signature mechanism. Do not
   compile or publish installable packages unless a repository explicitly
   overrides the source-snapshot policy. Downstream gates must consume the same
   ZIP bytes rather than recreating them.

## Decision

- `ready-to-tag` only when every applicable pre-tag hard gate passes and all
  unavailable evidence is intentionally deferred to an exact-tag hosted gate.
- `not-ready` when any required check fails or a correctness, security,
  integrity, compatibility, or coverage gap remains.
- `hold` when the candidate, comparison base, authority, profile, or required
  environment cannot be resolved safely.

Use `templates/bt6-release-readiness-report.md`. A `ready-to-tag` result becomes
stale on any candidate commit, dependency lock, policy, test, platform, or
release-configuration change. After tagging, always run `bt6-release-validation`
against the exact immutable tag; readiness is never release certification.
