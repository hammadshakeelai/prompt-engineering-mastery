# Tool-Augmented Language Models: Toolformer and Gorilla

## 1. Toolformer (Schick et al., 2023)
Self-supervised learning allowing LMs to teach themselves how/when to use external tools without human annotations.
- Candidate Generation: Model samples candidate API calls via few-shot prompts.
- Execution and Filtering: Calls external APIs (calculator, QA, Wikipedia, calendar, translation). Retains calls only if conditioning on API output reduces cross-entropy loss on future tokens.
- Finetuning: Fine-tunes base LM (GPT-J 6B) on filtered trajectories, learning autonomous tool invocation.

## 2. Gorilla (Patil et al., 2023)
LLaMA-based model fine-tuned to write correct executable API calls from natural language prompts.
- APIBench: Evaluated across 1600+ real-world APIs (TorchHub, HuggingFace, TensorHub). Outperforms GPT-4 and Claude in API generation accuracy.
- Generates complex multi-argument API requests while respecting schema constraints and API updates.

## 3. Retrieval-Aware Training (RAT)
Embeds retriever directly into the fine-tuning curriculum.
- Model conditioned on retrieved (and noisy/distractor) API docs during training.
- Forces LM to ground syntax and arguments in retrieved docs rather than stale parametric memory.
- Zero-shot generalization to unseen/modified APIs by updating the retriever index.