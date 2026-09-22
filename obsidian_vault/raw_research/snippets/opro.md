# OPRO: Optimization by PROmpting (Yang et al., DeepMind 2023)

- **LLM as Optimizer**: Derivative-free, black-box optimization defined entirely in natural language.
- **Meta-Prompting & Trajectory**: Guided by problem description + optimization trajectory (sorted past candidate solutions with scores). LLM recognizes trends and balances exploration/exploitation.
- **Results**: On GSM8K, boosted accuracy up to 8% (reaching 80.2% on PaLM 2-L with prompts like *"Take a deep breath and work on this problem step-by-step"*); up to 50% relative gain across BBH tasks.