# Speech-Native Latent Audio Steerability (Mimi, Moshi, SNAC & RVQ)

## 1. The Cascaded Audio Bottleneck
Traditional conversational voice agents cascade three disparate systems:
$$\text{Audio Input} \xrightarrow{\text{ASR}} \text{Text} \xrightarrow{\text{LLM}} \text{Text} \xrightarrow{\text{TTS}} \text{Audio Output}$$
- **Information Evaporation:** ASR transcription completely erases paralinguistic latent dynamics: pitch inflections, hesitation pauses, emotional timbre, vocal irony, and breathing.
- **Latency Compounding:** Each serialized pipeline stage incurs unchunked latency buffers, yielding interaction delays $>1.5\text{--}2.5\,\text{s}$ that prevent natural conversational turn-taking.

## 2. Neural Audio Codecs & Multi-Stream RVQ (Mimi / Moshi)
Kyutai's Moshi (Défossez et al., 2024) and Mimi codec pioneer end-to-end full-duplex speech modeling:
1. **Residual Vector Quantization (RVQ):**
   Continuous audio waveforms are compressed into discrete codebook indices across $Q$ quantization stages:
   $$z \approx \hat{z} = \sum_{q=1}^Q \mathbf{C}_q[i_q], \quad i_q \in \{1, \dots, K\}$$
2. **Semantic vs. Acoustic Disentanglement:**
   - In Mimi (operating at $12.5\,\text{Hz}$, $1.1\,\text{kbps}$), the first codebook stage ($q=1$) is aligned with a self-supervised speech representation (WavLM/HuBERT) capturing phonetic and linguistic semantics.
   - Codebooks $q \in [2, Q]$ capture acoustic residual features (speaker identity, reverberation, cadence, vocal timbre).
3. **Multi-Stream Full-Duplex Architecture:**
   The transformer autoregressively models interleaved user audio and agent audio streams simultaneously alongside inner text thought tokens, achieving $<200\,\text{ms}$ conversational latency.

## 3. Direct Latent Prosody Steerability
- **Codebook-Targeted Activation Steering:** Intervening in the residual stream corresponding to acoustic codebook positions ($q \ge 2$) directly steers emotional warmth, prosody, and speech rate without degrading semantic coherence ($q=1$).
- **Acoustic Backchanneling:** Natural interruptions ("mm-hmm", laughter, overlap) are modeled natively via conditioning tokens, bypassing explicit voice activity detection (VAD) state machines.
