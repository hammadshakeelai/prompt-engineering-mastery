<p align="center">
  <img src="./assets/banner.svg" alt="Prompt Engineering Mastery Banner" width="100%" />
</p>

<p align="center">
  <strong>The Definitive Knowledge Base & Autonomous Research Blueprint for Frontier Prompt Engineering</strong>
</p>

<p align="center">
  <a href="https://hammadshakeelai.github.io/prompt-mastery-app/"><img src="https://img.shields.io/badge/Live%20Web%20App-GitHub%20Pages-success.svg?style=for-the-badge&logo=github" alt="Live App"></a>
  <a href="https://github.com/hammadshakeelai/prompt-mastery-app/releases/latest"><img src="https://img.shields.io/badge/Download-Android%20APK-blue.svg?style=for-the-badge&logo=android" alt="Download APK"></a>
  <a href="#table-of-contents"><img src="https://img.shields.io/badge/Status-Active_Research_Brain-blue.svg?style=for-the-badge" alt="Status"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-emerald.svg?style=for-the-badge" alt="License"></a>
  <a href="https://github.com/hammadshakeelai"><img src="https://img.shields.io/badge/Author-hammadshakeelai-purple.svg?style=for-the-badge" alt="Author"></a>
</p>


---

## ⚡ Overview

**Prompt Engineering Mastery** has evolved from a static encyclopedia into an **Autonomous Research Brain**. It combines handcrafted, highly technical deep-dives with a live `obsidian_vault` populated by specialized subagents scouring the web for the absolute cutting edge (including system prompt leaks, ablation toolkits, and raw arXiv papers).

From tokenization mechanics and in-context learning theory to frontier test-time reasoning models (`o1`, `DeepSeek-R1`), programmatic optimization frameworks (`DSPy v2.5`, `TextGrad`), and adversarial security hardening (Many-Shot, Crescendo, Abliteration) — this repository serves as your architectural reference and autonomous research engine.

---

## 📱 Interactive Web Companion & Android App

Experience the entire 738-module curriculum through the dedicated client application:

* 🌐 **Live Web Application (GitHub Pages)**: [hammadshakeelai.github.io/prompt-mastery-app](https://hammadshakeelai.github.io/prompt-mastery-app/)
* 📦 **Native Android APK Release**: [Download `PromptMastery-v1.0.0.apk`](https://github.com/hammadshakeelai/prompt-mastery-app/releases/latest)
* 💻 **Client Source Repository**: [github.com/hammadshakeelai/prompt-mastery-app](https://github.com/hammadshakeelai/prompt-mastery-app)

**Key Features**:
* **100% Offline Capability**: Runs locally on mobile and desktop browsers via Workbox Service Worker or standalone Android APK.
* **In-Memory AI Tutor**: Instant Socratic explanations (< 2ms) without external API dependencies.
* **Audio Lecture Player**: Background audio synthesis for study on the go.
* **Zero Layout Collisions**: Screen-optimized for Samsung Galaxy S23 Ultra and all mobile viewports.

---


## 🧠 The Obsidian Vault (`obsidian_vault/`)

The repository contains an integrated **Obsidian Knowledge Graph** built by autonomous subagents.

* **`000_Master_Brain_Index.md`**: The central node mapping every paper, script, and theory file together.
* **`raw_research/OBLITERATUS/`**: Full clone of Elder Plinius's advanced LLM refusal ablation toolkit (Orthogonal Direction Modification).
* **`raw_research/chatgpt_system_prompts/`**: A curated extraction of the actual system prompts used by models like GPT-4o, Claude 3.5 Sonnet, and Apple Intelligence.
* **`raw_research/security_jailbreaks.md`**: Theoretical mechanics of Many-Shot Jailbreaking, Crescendo multi-turn attacks, token smuggling, and the Dual-LLM pattern.
* **`raw_research/programmatic_optimization.md`**: The latest APIs for DSPy MIPROv2, DSPy v2.5, TextGrad, and DeepMind's OPRO.
* **`raw_research/test_time_compute.md`**: Subagent research on test-time compute scaling laws, RLVR (Reinforcement Learning with Verifiable Rewards), and parsing `<think>` tokens.

---

## 🪝 Research Engine & Hooks (`research_engine/`)

To prevent tunnel vision and ensure rigorous architectural thinking, this repo includes meta-prompts meant to be injected into agent loops:
* **`critical_thinking_hook.md`**: An injection block that forces the AI to use an adversarial lens, question baseline assumptions, and map problems cross-disciplinarily (e.g., Control Theory, Compiler Design).

---

## 🧭 Curated Deep-Dive Modules

### 1. [Foundations & In-Context Mechanics](01_foundations/core_principles.md)
* **Autoregressive Generation Mechanics**: Logit scaling, temperature, stop sequences.
* **[In-Context Learning (ICL)](01_foundations/in_context_learning.md)**: Implicit gradient descent vs Bayesian inference, k-NN exemplar selection, Maximal Marginal Relevance (MMR).

### 2. [Reasoning Architectures](02_reasoning_architectures/cot_and_variants.md)
* **Linear**: Zero-Shot CoT, Few-Shot CoT, Least-to-Most decomposition, Step-Back abstraction.
* **[Non-Linear Search (ToT & GoT)](02_reasoning_architectures/tree_and_graph_of_thoughts.md)**: BFS/DFS beam exploration, DAG aggregations.
* **[Frontier Reasoning Models](02_reasoning_architectures/reasoning_models_frontier.md)**: Transitioning to outcome-oriented constraints for `o1`/`DeepSeek-R1`.

### 3. [Agentic Systems & Tool Calling](03_agentic_and_tool_prompting/react_and_autonomous_loops.md)
* **Control Loops**: ReAct, Plan-and-Solve, Reflexion, Language Agent Tree Search (LATS).
* **[Structured Outputs](03_agentic_and_tool_prompting/structured_outputs_and_tool_calling.md)**: Constrained Context-Free Grammar (CFG) decoding vs prompt instructions.
* **[Multi-Agent Coordination](03_agentic_and_tool_prompting/multi_agent_prompting.md)**: Supervisor-worker hierarchies, adversarial debate.

### 4. [Programmatic & Automated Optimization](04_programmatic_and_automated_optimization/dspy_deep_dive.md)
* **[DSPy Deep Dive](04_programmatic_and_automated_optimization/dspy_deep_dive.md)**: Signatures, Modules, and MIPROv2 Bayesian prompt compilation.
* **[Automatic Prompt Engineering (APE)](04_programmatic_and_automated_optimization/meta_prompting_and_ape.md)**: Monte Carlo proposal distributions, PromptBreeder.
* **[TextGrad & Gradient-Free Search](04_programmatic_and_automated_optimization/textgrad_and_gradient_free_opt.md)**: Textual backpropagation, natural language loss functions.

---

## 🤖 Automated Scripts (`scripts/`)
* **`paper_downloader.py`**: An open-source PDF fetcher using the `arxiv` python pip package to automatically gather academic papers on prompt engineering into the vault.
* **`build_obsidian_index.py`**: A python script that crawls the entire repo and regenerates the interactive `000_Master_Brain_Index.md` Graph.

---

## 🛠️ Getting Started

Clone the repository and explore the research guides:

```bash
git clone https://github.com/hammadshakeelai/prompt-engineering-mastery.git
cd prompt-engineering-mastery
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
