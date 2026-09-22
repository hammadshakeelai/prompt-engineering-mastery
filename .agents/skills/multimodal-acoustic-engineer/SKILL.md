---
name: multimodal-acoustic-engineer
description: Specialized directive for speech-native audio modeling (Mimi, Moshi, SNAC, RVQ codecs), interleaved audio-text generation, direct acoustic/prosody steering, and zero-cascading real-time speech dialogue.
---

# Multimodal Acoustic Engineer Skill

Use this skill when designing speech-native foundation model interactions, analyzing discrete neural audio codecs (RVQ, SNAC, Mimi), steering acoustic properties (prosody, latency, emotion, turn-taking), or architecting zero-cascading voice agents.

## 1. Speech-Native Mechanics vs. Cascaded Pipelines

1. **Elimination of the Cascaded Bottleneck:**
   - Traditional voice systems use a 3-stage cascade: Automatic Speech Recognition (ASR) $\to$ Text LLM $\to$ Text-to-Speech (TTS).
   - This cascade discards all paralinguistic nuance (pitch, cadence, hesitation, laughter, emotion) at the ASR stage and introduces compounding latency ($>1.5\,\text{s}$).
   - Speech-native architectures (e.g., Moshi / Mimi, GPT-4o voice mode) tokenize audio directly into discrete acoustic/semantic tokens, enabling end-to-end full-duplex inference with theoretical latency $<200\,\text{ms}$.

2. **Residual Vector Quantization (RVQ) Dynamics:**
   - Speech waveforms are compressed using neural autoencoders with multi-stage RVQ codebooks:
     $$e = \sum_{q=1}^Q c_q(i_q)$$
   - **Semantic vs. Acoustic Partitioning:** In Mimi, low-depth codebooks capture semantic/phonemic identity (shared with text representations), while higher-depth codebooks capture acoustic texture, acoustic environment, and speaker timbre.

## 2. Full-Duplex Dialogue & Turn-Taking Steerability

1. **Interleaved Multistream Decoding:**
   - Models process user audio streams and agent audio streams simultaneously across dual channels.
   - Steer conversational turn-taking by conditioning on silent tokens, backchannel acoustic cues ("mm-hmm", "yeah"), and barge-in interruptions without explicit state machines.

2. **Acoustic Latent Steering:**
   - Rather than relying solely on descriptive text system prompts ("Speak in an energetic tone"), steer prosody directly through conditioning tokens or activation interventions in early audio codebook layers.
