# Activation Steering via Steering Vectors (CAA & Refusal Clamping)

- **Contrastive Activation Addition (CAA)**: Isolates behavioral directions by computing the mean difference between activations elicited by contrasting prompt pairs (e.g., compliant vs. refusing). Adding this vector during forward passes dynamically steers behavior.
- **Refusal Vector Clamping**: Targets the linear subspace governing safety refusals. Clamping or projecting out activations along this vector neutralizes refusal mechanisms without degrading general capabilities.