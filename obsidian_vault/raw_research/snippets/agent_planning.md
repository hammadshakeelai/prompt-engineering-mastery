# Agent Planning and Task Decomposition

## Taxonomy & Core Mechanisms (Huang et al., 2024 Survey)
Huang et al. (2024) categorize LLM agent planning into task decomposition, plan selection, external modules, reflection, and memory:
- **Subgoal Generation:** Decomposes high-level goals into DAGs or linear action sequences, dynamically adapting steps based on environment feedback.
- **Hierarchical Task Networks (HTN):** Recursively breaks compound tasks into intermediate subtasks and primitive actions, enabling multi-level abstraction.
- **PDDL-Style Neuro-Symbolic Planning:** LLM translates unstructured descriptions into formal Planning Domain Definition Language (PDDL), allowing sound symbolic solvers to guarantee plan validity.

## Key Foundational Frameworks
- **Inner Monologue (Huang et al., 2022):** Pioneer of closed-loop embodied planning. Integrates multimodal environmental feedback (success detection, scene descriptions) into continuous language prompts for dynamic re-planning without fine-tuning.
- **Voyager (Wang et al., 2023):** Lifelong embodied learning agent in Minecraft:
  1. *Automatic Curriculum:* Progressively generates open-ended subgoals tailored to current state.
  2. *Skill Library:* Synthesizes and self-refines reusable executable JavaScript routines, indexed via vector embeddings.
  3. *Iterative Prompting:* Incorporates compiler errors and environmental feedback for self-correction.