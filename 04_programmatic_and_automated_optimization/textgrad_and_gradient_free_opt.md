# 03. TextGrad & Gradient-Free Prompt Optimization

TextGrad (Yuksekgonul et al., Stanford, 2024) and OPRO (Yang et al., Google DeepMind) formalize automated prompt optimization by extending the concept of **backpropagation** to natural language.

---

## 1. TextGrad: "Backpropagation" via Natural Language Feedback

In classical deep learning, a scalar loss $\mathcal{L}$ backpropagates gradients $\frac{\partial \mathcal{L}}{\partial W}$ via the chain rule to update numerical weights:

$$W \leftarrow W - \eta \nabla_W \mathcal{L}$$

TextGrad recognizes that in compound AI systems:
- Variables are **strings** (prompts, code, thoughts).
- Operations are **LLM forward calls** or tool executions.
- Gradients are **natural language critiques and directional corrections** returned by LLM evaluators.

```
[System Prompt (Variable)] ---> [LLM Forward Pass] ---> [Response (Variable)]
             ^                                                  |
             | Textual Gradient                                 v
             | (LLM critique & update step)            [Loss Function / Critic]
             +--------------------------------------------------+
```

### The TextGrad Computation Graph:
1. **Forward Pass**: Prompt $P$ processes input $x$ to yield completion $y = \text{LLM}(P, x)$.
2. **Loss Evaluation**: A Critic LLM evaluates $y$ against criteria (e.g. correctness, conciseness, safety), outputting textual critique $g_y$.
3. **Backward Pass**: The gradient $g_y$ is propagated back to prompt $P$. The TextGrad Optimizer prompts an LLM:
   `"Here is prompt P. It produced output y. The evaluation critique was g_y. Rewrite P to rectify this defect."`
4. **Step**: Prompt $P$ is updated: $P_{t+1} \leftarrow \text{Optimizer}(P_t, g_y)$.

---

## 2. TextGrad in Python

```python
import textgrad as tg

# 1. Initialize language model and system prompt variable
llm = tg.get_engine("gpt-4o")
system_prompt = tg.Variable(
    "You are an assistant answering complex physics questions.",
    requires_grad=True,
    role_description="System prompt for physics QA"
)

# 2. Build computation graph
model = tg.BlackBoxLLM(llm, system_prompt=system_prompt)
loss_fn = tg.TextLoss("Evaluate if the explanation is intuitive, mathematically rigorous, and avoids jargon.")
optimizer = tg.TGD(parameters=[system_prompt], engine=llm)

# 3. Training Loop
for epoch in range(num_epochs):
    for x, y_gold in dataloader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y_gold)
        loss.backward()        # Propagates textual critiques backwards
        optimizer.step()       # Updates system_prompt text
```

---

## 3. OPRO: Optimization by PROmpting (Google DeepMind)

OPRO (Yang et al., 2023) uses an LLM as an optimizer across numerical benchmarks (GSM8K, Big-Bench Hard) and linear regression:
- Tracks history of past candidate prompts paired with their validation scores:
  `Score: 78.4% | Prompt: "Let's work through this step by step."`
  `Score: 84.1% | Prompt: "Take a deep breath and break this down into atomic calculations."`
- Prompts the LLM optimizer to infer which linguistic traits correlate with higher scores and synthesize a novel candidate that surpasses the current maximum.
