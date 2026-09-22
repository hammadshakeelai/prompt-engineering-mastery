# Prompt Injection and Indirect Prompt Injection Attacks

## 1. Mechanical Attack Vector
Exploits conflation of control plane (system instructions) and data plane (user/retrieved text).
Both serialized into one autoregressive sequence; transformer cannot distinguish meta-instructions from passive data.
Adversarial tokens mimicking authoritative syntax hijack model's objective function at runtime.

## 2. Greshake et al. (2023) Findings
Paper: "Not what you've signed up for" (arXiv:2302.12173)
Formalized Indirect Prompt Injection (IPI): attackers embed payloads in external resources (webpages, emails, APIs) retrieved by the agent.
Demonstrated: silent data exfiltration via Markdown image rendering, arbitrary tool execution, session contamination, self-replicating prompts (LLM worms).

## 3. Indirect Injection via Web Content
When agents scrape web content, hidden DOM elements (display:none) or HTML comments embed instructions.
Upon ingestion, payload overrides initial system goal, coercing model to leak PII or invoke destructive tools.

## 4. Defensive Patterns
- Input Sanitization: XML/JSON structural delimitation, nonces/spotlighting, perimeter classifiers (SecAlign, Llama Guard).
- Privilege Separation (Dual-LLM): Quarantined LLM parses untrusted data, returns non-executable content; separate Privileged LLM retains tool/API access.