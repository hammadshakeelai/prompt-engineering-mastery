# Many-Shot Jailbreaking (Anthropic, 2024)

- **Mechanism**: Prepends dozens to hundreds of faux dialogues where an assistant compliantly answers harmful requests, conditioning the model via in-context learning (ICL) saturation.
- **Scaling Dynamics**: Attack success rates follow power-law scaling as demonstrations scale (64 to 256+ shots), overriding RLHF safety guardrails.
- **Defenses**: Context truncation, perimeter classifiers, and alignment fine-tuning specifically tailored against long-context adversarial formatting.