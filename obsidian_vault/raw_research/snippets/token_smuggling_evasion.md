# Token Smuggling & Encoding Evasion Attacks

- **Mechanism**: Obfuscates restricted instructions using alternative representations (Base64, ROT13, Morse code, emoji ciphers) to bypass perimeter regex/moderation filters.
- **Vulnerability**: Sufficiently capable LLMs decode and reconstruct representations internally, executing forbidden payloads.
- **Defenses**: Multi-pass decoding prior to safety classification and token normalization.