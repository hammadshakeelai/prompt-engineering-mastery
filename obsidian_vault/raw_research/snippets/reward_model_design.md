# Reward Model Design for RLHF: Process vs. Outcome

## ORMs vs. PRMs
- Outcome Reward Models (ORMs): Evaluate only the terminal output. Simple annotation, but process-blind - reward correct final answers even via flawed intermediate reasoning.
- Process Reward Models (PRMs): Step-level feedback scoring each individual reasoning step. Granular credit assignment, better for MCTS/Best-of-N search. Require high annotation overhead.

## PRM800K Dataset (Lightman et al. 2023 - OpenAI "Let's Verify Step by Step")
- 800,000 human step-level ratings across 75,000 solutions on MATH benchmark.
- Proved process supervision substantially outperforms outcome supervision for complex reasoning.
- Foundation for step-level verification showing denser supervision produces more consistent reasoning.

## Reward Model Quality as RLHF Bottleneck
- Goodhart's Law: RL optimization exploits RM blind spots (length, format, sycophancy) rather than acquiring true capabilities.
- Distributional Shift: As policy drifts OOD, RMs output uncalibrated heavy-tailed positive scores.
- Objective Compression: Collapsing multidimensional human preferences into scalars discards vital nuance.