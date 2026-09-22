# Model Context Protocol (MCP, Anthropic 2024–2025)

## 1. The $N \times M$ Tool Integration Bottleneck
Prior to late 2024, integrating LLMs with developer tools, local file systems, databases, and enterprise APIs required bespoke API connectors for every model and framework pairing ($N$ models $\times M$ tools). 

Anthropic released the **Model Context Protocol (MCP)** in November 2024 as an open-standard client-host-server protocol that unifies agent tool invocation, context injection, and resource access.

## 2. Protocol Architecture & Primitives
MCP defines a three-tier architecture: **Host** (e.g., Claude Desktop, IDE), **Client** (protocol adapter inside host), and **Server** (lightweight process exposing resources and tools). Communication is conducted via **JSON-RPC 2.0**:

- **Tools (`tools/list`, `tools/call`):** Executable functions that models invoke with strict JSON schemas. Results return text or binary artifacts.
- **Resources (`resources/list`, `resources/read`, `resources/subscribe`):** URI-addressable contextual data (files, database tables, live telemetry) that clients attach to context. Supports push notifications when resources update.
- **Prompts (`prompts/list`, `prompts/get`):** Server-defined prompt templates and conversational workflows with dynamic arguments.
- **Sampling (`sampling/createMessage`):** Allows MCP servers to request LLM generations back through the client host, enabling autonomous sub-agent delegation under host guardrails.

## 3. Transport Mechanisms & Security Isolation
- **stdio Transport:** Local execution where the client launches server subprocesses, communicating through standard I/O pipes (`stdin`/`stdout`). Eliminates network port exposure.
- **Streamable HTTP (SSE):** Remote network transport using Server-Sent Events for streaming server-to-client notifications and HTTP POST for client-to-server requests.
- **Least-Privilege Sandboxing:** Clients maintain explicit authorization boundaries; tool executions and file reads require explicit host-level approval or scoped capability tokens.
