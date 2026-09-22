# System Prompt Architecture Patterns

An analysis of official system prompts from industry-leading AI assistants (Claude, Cursor, v0, Grok, Replit).

## 1. Persona Definition
- **Clear Role-Playing:** Models are given explicit identities and roles. For example, Cursor acts as a "pair programmer," Replit as an "expert autonomous programmer," and v0 as "Vercel's AI-powered assistant."
- **Tone and Attitude:** Prompts strictly regulate tone. Claude is instructed to be helpful, honest, harmless, and to avoid being preachy. Replit must use "simple, everyday language" for non-technical users.

## 2. Capabilities & Behaviors
- **Context Awareness:** Prompts define what context the model receives (e.g., Cursor is told it receives lint errors, open files, and terminal state).
- **Tool Mastery:** Complex orchestration is defined. Claude uses a decision tree for search complexity (from 0 up to 20 tool calls). v0 leverages MDX to output executable React environments and Node.js blocks.
- **Autonomy:** Coding agents like Replit and Cursor are instructed to autonomously execute tasks, verify functionality (e.g., using Replit's web application feedback tool), and iterate before asking for user input.

## 3. Limitations & Constraints
- **Knowledge Cutoffs:** Claude is explicitly told its cutoff date (Jan 2025) and how to handle current events.
- **Operational Boundaries:** Replit forbids the use of Docker/virtual environments and insists on Replit workflows. v0 forbids `package.json` creation and `require` statements.
- **Strict Overrides:** Grok is limited to only editing images it previously generated. Cursor is constrained to a maximum of 3 loops for fixing linter errors before asking the user.

## 4. Tool Access and Usage
- **Schema & Formatting:** Tool schemas are often deeply embedded in the prompt. Cursor provides exact JSON schemas for its tools.
- **Explanation Requirements:** Cursor must explain *why* it is using a tool before calling it.
- **Targeted Abstractions:** v0 uses React components like `<CodeProject>`, `<QuickEdit />`, and `<AddEnvironmentVariables>` as native tools. Replit uses `str_replace_editor` and `search_filesystem`.

## 5. Output Formatting
- **XML/Tagging:** Heavy reliance on XML tags for internal reasoning (`<Thinking>`, `<rationale>`) and structured outputs (`<CodeProject>`, `// ... existing code ...` diffs).
- **Code Citations:** Cursor demands a precise format for code blocks (`startLine:endLine:filepath`).
- **Modularity:** Claude is directed to use "Artifacts" for substantial UI/code, while v0 uses `type="code"` attributes on markdown code blocks for special rendering.

## 6. Safety Guardrails & Compliance
- **Copyright strictness:** Claude has an extensive `<mandatory_copyright_requirements>` section, strictly forbidding displacive summaries and limiting quotes to under 15 words.
- **Harmful Content:** Standard safety guidelines against generating weapons, malware, or exploiting minors. v0 specifies providing a standard refusal without explanation or apology.
- **Data Integrity:** Replit explicitly mandates "Data Integrity," requiring the model to ask for API keys rather than spoofing fake data.
- **Destructive Actions:** Replit forbids destructive SQL statements (DELETE/UPDATE) unless explicitly asked. Cursor requires user approval before terminal command execution.
