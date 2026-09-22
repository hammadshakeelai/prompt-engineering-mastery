# Speech-Native Prompting & Prosody Steering in Audio Language Models (2024–2026)

## 1. Paradigm Shift: Cascaded vs. Native Audio Tokenization
Traditional voice systems relied on a cascaded tripartite pipeline: Automatic Speech Recognition (ASR) $\to$ Text LLM $\to$ Text-to-Speech (TTS). This pipeline discarded non-verbal acoustic signals (inflection, hesitation, pitch, emotional stress) and accumulated latency across stages (>1,500ms).
**Audio Language Models (ALMs)**—such as Gemini 2.0/2.5 Flash Native Audio and GPT-4o voice—unify modalities natively. Continuous audio waveforms are quantized into discrete neural acoustic/semantic codec tokens interleaved directly in the autoregressive context window, enabling sub-300ms duplex conversational loops.

## 2. Prosody Directing: Prompting as Voice Performance
Because native ALMs interpret and generate speech directly, prompt engineering shifts from text drafting to **vocal directing**:
- **Speaker Persona & Acoustic Scaffolding:** Prompts specify vocal timbre, pacing, cadence, and ambient environment (e.g., *"Speak in a relaxed, warm tone suitable for a quiet bedroom"*).
- **Inline Expressive Tags:** Models support inline directive bracket tags within conversational prompts to modulate prosody token-by-token:
  - Emotional markers: `[enthusiastic]`, `[sympathetic]`, `[calm]`
  - Paralinguistic phonations: `[whispers]`, `[sighs]`, `[laughs]`, `[clears throat]`
  - Pacing delimiters: `[pause: 1s]`, `[hurriedly]`

## 3. Affective Dialogue & Bidirectional Acoustic Interpretation
Native ALMs perform bidirectional prosody adaptation:
- **Acoustic Cue Interpretation:** The model extracts intent from the user's speech velocity, pitch variance, and vocal stress (e.g., detecting user distress or background noise) rather than relying solely on semantic words.
- **Dynamic Prosodic Matching:** The model matches response urgency and pitch contour to the conversational context.

## 4. Evaluation: Live-ProsodyJudge (LPJ)
Evaluating native voice generation bypasses text BLEU/ROUGE in favor of multimodal benchmarks:
- **Live-ProsodyJudge (LPJ):** Evaluates streaming intonation accuracy, turn-taking latency, natural interruption handling, and emotional fidelity against human perceptual rubrics.
