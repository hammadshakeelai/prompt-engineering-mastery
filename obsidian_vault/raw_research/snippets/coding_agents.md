# Agentic AI Coding Systems: SWE-agent, Devin, and OpenHands (2024)

## Architecture & Problem Solving
Autonomous coding agents resolve repository-level GitHub issues through iterative ReAct/CodeAct loops:
1. **Context & Localization**: Ingesting issue descriptions and searching codebases to isolate fault locations.
2. **Reproduction & Verification**: Generating reproducing scripts, applying edits, and executing test suites to verify regressions before finalizing git patches.

Systems diverge in architecture: **SWE-agent** pioneered dedicated Agent-Computer Interfaces (ACI) for LM-centric navigation; **Devin** introduced a managed environment with shell, browser, and persistent planner; **OpenHands** established open-source Docker sandboxing using executable Python/bash (CodeAct).

## Tool Use Patterns
- **Search & Navigation**: High-level commands (`find_file`, `search_dir`, ripgrep) return condensed, token-budgeted directory trees to prevent context overflow.
- **Code Editor / ACI**: Windowed viewers (~100 lines with scrolling/offset) paired with surgical line-replacement or pattern-matching editors, pre-validated via linting.
- **Bash & Sandboxed Execution**: Persistent Linux shells run commands and invoke pytest. **Claude 3.5 Sonnet Computer Use** extended this to OS GUIs, terminals, and browsers via screen perception.

## Performance on SWE-Bench (2024)
- Early 2024: Raw LLMs scored <2-4%. Devin reported **13.86%** unassisted resolve rate; SWE-agent achieved **~12.5%** (Full) and **~18%** (Lite) with GPT-4.
- Late 2024 (SWE-bench Verified): Claude 3.5 Sonnet hit **49.0%**, while **OpenHands (CodeAct v2.1)** achieved **53.0%**.