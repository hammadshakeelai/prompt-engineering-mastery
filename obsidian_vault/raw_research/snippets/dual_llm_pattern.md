# Dual-LLM Pattern for Indirect Prompt Injection Defense

1. **Quarantined LLM**: Ingests untrusted external data (web pages, emails, third-party files) to extract or parse information. Has NO access to sensitive tools, private context, or actuators.
2. **Privileged LLM**: Interacts with user, maintains private context, and controls sensitive tools/APIs. Never reads raw untrusted input, receiving only sanitized structured outputs from the Quarantined LLM.
Mitigates indirect prompt injection by enforcing strict trust and capability boundaries.