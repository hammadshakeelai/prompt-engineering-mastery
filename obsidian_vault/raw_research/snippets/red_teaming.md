# Constitutional AI Red-Teaming & Automated Adversarial Testing

## Perez et al. (2022) & Automated Generation
Perez et al. (2022) pioneered using language models to red-team other language models (*"Red Teaming Language Models with Language Models"*). A generator LM crafts adversarial prompts to surface toxic outputs, private data leakage, and harmful edge cases.
- **Zero-shot / Few-shot Generation:** Prompting the red-team LM with instructions or seeded examples of harmful queries.
- **Reinforcement Learning (RL):** Training the adversary model with RL to maximize harmfulness scores produced by a safety classifier while applying a KL penalty to preserve fluency.

## Anthropic's Red Team Findings
- **Scaling Behaviors (Ganguli et al. 2022):** Larger, RLHF-aligned models become substantially harder to jailbreak via naive prompts, demanding more complex, multi-turn adversarial attacks.
- **Vulnerability Patterns:** Models exhibit sycophancy, persona adoption biases, and subtle safety degradation across multi-turn interactions.
- **Constitutional AI Integration (Bai et al. 2022):** Models generate adversarial prompts, produce responses, critique them against constitutional principles, and revise them (RLAIF).

## Automated vs. Human Red-Teaming
- **Throughput & Coverage:** Automated testing generates hundreds of thousands of diverse attacks rapidly, uncovering long-tail failure modes and boundary vulnerabilities.
- **Depth vs. Breadth:** Human red-teamers excel at nuanced social engineering and zero-day conceptual jailbreaks; automated testing excels at systematic regression testing.