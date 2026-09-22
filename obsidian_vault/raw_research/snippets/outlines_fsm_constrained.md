# Outlines: FSM Grammar-Constrained Decoding

- **Compilation**: Compiles Context-Free Grammars (CFGs) and JSON Schemas into a Finite State Machine (FSM).
- **Logit Masking**: At each token generation step, sets logits of syntactically invalid tokens to $-\infty$ before softmax:
  $$z'_{t, i} = \begin{cases} z_{t, i} & \text{if valid} \\ -\infty & \text{otherwise} \end{cases}$$
- **Guarantee**: Eliminates JSON syntax and parsing errors in a single forward pass with zero retry latency.