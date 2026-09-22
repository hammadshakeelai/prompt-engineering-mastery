# Vision-Language-Action (VLA) Prompting & Diffusion Policies in Robotics (2024–2026)

## 1. The Vision-Language-Action (VLA) Paradigm
Traditional robot learning trained isolated imitation learning or reinforcement learning policies per task and embodiment. Vision-Language-Action (VLA) models unify perception, semantic comprehension, and motor control by training autoregressive or diffusion models on the multi-robot **Open X-Embodiment** corpus.
Natural language prompts act directly as goal conditioners, mapping high-level instructions into low-level joint velocities, end-effector poses, and gripper states.

## 2. Architectural Comparison: OpenVLA vs. Octo
- **OpenVLA (Kim et al., 2024):** Integrates a 7B LLaMA-2 backbone with dual vision encoders (DINOv2 for spatial geometry + SigLIP for semantic alignment). Continuous 7-DoF robot actions are discretized into 256 uniform bins per dimension and emitted directly as discrete action tokens, unifying text generation and physical actuation within a single token vocabulary.
- **Octo (Octo Model Team, 2024):** Employs a modular Transformer backbone paired with a **Conditional Diffusion Action Head**. Continuous robot trajectories are generated via iterative denoising conditioned on text embeddings and camera patch tokens, producing smooth, multi-modal continuous action distributions.

## 3. Prompting Dynamics for Embodied Control
1. **Spatial Disambiguation:** Unlike conversational LLMs, VLA policies require explicit spatial grounding. Prompts specifying relative spatial frames (*"pick up the red mug by its handle to the right of the bowl"*) prevent multi-modal attractor collapse.
2. **Object Affordance Guidance:** Describing functional affordances (*"grasp the bottle from the neck"*) directly steers the diffusion policy's basin of attraction toward kinematically stable grasps.
3. **Hierarchical Task Decomposition:** High-level goals (*"clean the kitchen"*) fail if fed directly to low-level VLA policies. Frontier agent scaffolds use high-level LLM planners (e.g., Voyager, Inner Monologue) to decompose long horizons into atomic primitive directives dispatched sequentially to the VLA policy.
