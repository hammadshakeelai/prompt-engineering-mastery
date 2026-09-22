# Edge Attribution Patching with Integrated Gradients (EAP-IG, Hanna et al. 2024)

## 1. Exhaustive Patching Bottleneck
Mechanistic circuit discovery relies on causal activation patching, swapping activation $h_u$ between clean ($x_{\text{clean}}$) and corrupted ($x_{\text{corrupted}}$) runs. Exhaustively measuring all edge interventions $u \to v$ requires $O(|E|) \sim 10^5\text{--}10^7$ forward passes, making it intractable for large models.

## 2. Base EAP & Gradient Saturation
Syed et al. (2023) introduced **Edge Attribution Patching (EAP)** via first-order Taylor expansion:
$$\Delta \mathcal{L}_{u \to v} \approx (h_u^{\text{clean}} - h_u^{\text{corrupted}})^\top \nabla_{h_u} \mathcal{L}(x_{\text{clean}})$$
Computes all edge attributions in a single backward pass. However, standard EAP fails due to **gradient saturation**: in saturated activation regimes (flat ReLU or extreme Softmax), $\nabla \mathcal{L} \approx 0$, falsely discarding causally vital edges.

## 3. Path-Integrated EAP-IG (Hanna et al. 2024)
Integrates gradients along the linear interpolation path between corrupted and clean activations:
$$\Delta \mathcal{L}_{u \to v}^{\text{IG}} = (h_u^{\text{clean}} - h_u^{\text{corrupted}})^\top \int_0^1 \nabla_{h_u} \mathcal{L}\left( h_u^{\text{corrupted}} + \alpha (h_u^{\text{clean}} - h_u^{\text{corrupted}}) \right) d\alpha$$
- **Axiomatic Completeness:** Satisfies $\sum_e \Delta \mathcal{L}_e^{\text{IG}} = \mathcal{L}(x_{\text{clean}}) - \mathcal{L}(x_{\text{corrupted}})$.
- **Faithfulness:** Discovers circuits with $>95\%$ behavioral fidelity to the uncompressed network while running $1000\times$ faster than exhaustive causal interventions.
