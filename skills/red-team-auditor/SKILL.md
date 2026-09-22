---
name: red-team-auditor
description: Auditing, testing, and hardening LLM prompts, agent workflows, and API integrations against jailbreaks, prompt injections, data exfiltration, and model evasion.
---

# Red Team Auditor Skill

Use this skill to evaluate and harden prompts, tools, and autonomous agent loops against adversarial manipulation, indirect prompt injection, and safety failures.

## 1. Threat Modeling Dimensions

1. **Direct Prompt Injection (DPI)**:
   - Malicious user prompts attempting to override system directives or persona boundaries.
   - Mitigation: XML delimiter scoping, instruction hierarchy enforcement (developer role precedence), and defensive prefix/suffix sandwiching.

2. **Indirect Prompt Injection (IPI)**:
   - Untrusted third-party content (scraped webpages, ingested emails, database entries) embedding adversarial instructions.
   - Mitigation: Implement the **Dual-LLM Pattern** (Quarantined LLM for data extraction with zero tool access; Privileged LLM for decision making).

3. **Many-Shot & In-Context Saturation**:
   - Long-context conditioning with repeated faux dialogues that overwhelm safety alignment.
   - Mitigation: Context length monitoring, input classification filters, and alignment fine-tuning.

4. **Token Smuggling & Encoding Evasion**:
   - Obfuscated payloads (Base64, ROT13, Morse, multi-lingual fragments) bypassing perimeter regex/keyword filters.
   - Mitigation: Canonical input normalization and multi-pass pre-classification.

5. **Activation-Level Steerability & Abliteration**:
   - Understanding internal representation vectors (refusal directions) and assessing model robustness when internal weights or activations are manipulated.
