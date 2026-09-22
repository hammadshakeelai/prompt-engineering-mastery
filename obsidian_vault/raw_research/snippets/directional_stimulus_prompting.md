# Directional Stimulus Prompting (Li et al. 2023)

- **Dual-Model Scaffolding**: A small, lightweight policy model generates discrete hints or directional stimuli (key concepts, keywords) appended to prompts for a frozen large LLM.
- **RL Optimization**: Policy model is trained via reinforcement learning using downstream LLM rewards, steering generation without large-model parameter updates.