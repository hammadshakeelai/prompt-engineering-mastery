import os

base = r'c:\Users\HP\Documents\antigravity\vibrant-kepler\obsidian_vault\raw_research\snippets'
os.makedirs(base, exist_ok=True)

files = {
'parent_document_retriever.md': """# Parent Document Retrieval / Small-to-Big Chunking

- **Mechanism**: Splits documents into small child chunks (sentences/paragraphs) mapped to larger parent chunks or full documents.
- **Search**: Matches user queries against fine-grained child embeddings for high precision.
- **Expansion**: Replaces child chunks with full parent context prior to passing into LLM context, avoiding semantic fragmentation.""",

'token_smuggling_evasion.md': """# Token Smuggling & Encoding Evasion Attacks

- **Mechanism**: Obfuscates restricted instructions using alternative representations (Base64, ROT13, Morse code, emoji ciphers) to bypass perimeter regex/moderation filters.
- **Vulnerability**: Sufficiently capable LLMs decode and reconstruct representations internally, executing forbidden payloads.
- **Defenses**: Multi-pass decoding prior to safety classification and token normalization.""",

'outlines_fsm_constrained.md': """# Outlines: FSM Grammar-Constrained Decoding

- **Compilation**: Compiles Context-Free Grammars (CFGs) and JSON Schemas into a Finite State Machine (FSM).
- **Logit Masking**: At each token generation step, sets logits of syntactically invalid tokens to $-\\infty$ before softmax:
  $$z'_{t, i} = \\begin{cases} z_{t, i} & \\text{if valid} \\\\ -\\infty & \\text{otherwise} \\end{cases}$$
- **Guarantee**: Eliminates JSON syntax and parsing errors in a single forward pass with zero retry latency.""",

'representation_engineering_repe.md': """# Representation Engineering (RepE, Zou et al. 2023)

- **Concept Vectors**: Isolates high-level directions in activation space (honesty, deception, emotion) via contrastive prompt pairs and PCA / difference-in-means across intermediate layers.
- **Reading**: Probes internal states to monitor latent deception or bias during generation.
- **Control**: Steers model behavior at inference by adding or subtracting scaled concept vectors from hidden activations without fine-tuning.""",

'cursor_system_prompt_design.md': """# Cursor IDE System Prompt Design for Code Agents

- **Two-Tier Diff Architecture**: Avoids raw code block output; uses surgical file-editing tools with `// ... existing code ...` placeholders, offloading patch application to an ultra-fast secondary model.
- **Bounded Autonomy**: 3-iteration limit on fixing linter errors before prompting user; enforces user approval for terminal execution and file citation boundaries (`startLine:endLine:filepath`).""",

'sglang_radix_jumpforward.md': """# SGLang RadixAttention & Jump-Forward Decoding

- **RadixAttention**: Organizes cached KV tokens into a radix tree, automatically reusing KV cache across prompts sharing prefixes (system instructions, multi-turn dialogue, few-shot examples) via LRU eviction.
- **Jump-Forward Decoding**: Identifies deterministic tokens in structured grammars (JSON syntax, keys) and jumps forward by emitting parallel token batches, bypassing redundant sequential forward passes.""",

'transformerlens_activation_patching.md': """# TransformerLens: Residual Stream Interventions & Causal Tracing

- **Residual Stream Interventions**: Hooks into `resid_pre` and `resid_post` to read, steer, ablate, or overwrite activation vectors at specific layers and token positions.
- **Activation Patching**: Swaps activations between clean and corrupted forward passes to establish causal links, measuring sufficiency (denoising) and necessity (noising) to isolate functional circuits."""
}

written = 0
for fname, content in files.items():
    path = os.path.join(base, fname)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    written += 1

print(f'Wrote {written} snippet files.')
