import os
import re
import glob
import json

KEPLER_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
APP_DIR = os.path.abspath(os.path.join(KEPLER_DIR, '..', 'prompt-mastery-app'))
OUTPUT_CURRICULUM_PATH = os.path.join(APP_DIR, 'src', 'data', 'curriculum.ts')

print(f"Kepler Directory: {KEPLER_DIR}")
print(f"Target Curriculum Path: {OUTPUT_CURRICULUM_PATH}")

TRACK_DEFINITIONS = [
    {
        'id': 'track-1-primer',
        'title': 'Track 1: Master Foundations & Core Principles Primer',
        'tagline': 'Generation mechanics, architectural prompt anatomy, few-shot ICL, ReAct loops, DSPy, and TextGrad.',
        'iconName': 'BookOpen',
        'color': 'from-emerald-500 to-teal-700',
    },
    {
        'id': 'track-2-icl-physics',
        'title': 'Track 2: The Physics of In-Context Learning & Induction Circuits',
        'tagline': 'Induction circuits, implicit meta-gradients, Bayesian task selection, glitch tokens, and embedding geometry.',
        'iconName': 'Cpu',
        'color': 'from-teal-500 to-emerald-700',
        'keywords': ['induction', 'in-context', 'gradient descent', 'bayesian', 'glitch', 'embedding', 'tokenization', 'physics', 'transformer', 'icl']
    },
    {
        'id': 'track-3-reasoning-topologies',
        'title': 'Track 3: Foundational & Structured Prompting Topologies',
        'tagline': 'CoT, Least-to-Most, Skeleton-of-Thought (SoT), Thread-of-Thought (ThoT), and Step-Back abstraction.',
        'iconName': 'Layers',
        'color': 'from-teal-500 to-cyan-700',
        'keywords': ['chain-of-thought', 'cot', 'skeleton', 'step-back', 'thread-of-thought', 'least-to-most', 'buffer-of-thoughts', 'few-shot', 'system prompt']
    },
    {
        'id': 'track-4-test-time-compute',
        'title': 'Track 4: Deliberate Reasoning Search & Test-Time Compute (TTC)',
        'tagline': 'Tree/Graph of Thoughts, MCTS, Process Reward Models (PRMs), RLVR, and Critic-Free GRPO (DeepSeek-R1).',
        'iconName': 'GitBranch',
        'color': 'from-blue-500 to-indigo-700',
        'keywords': ['tree of thoughts', 'graph of thoughts', 'mcts', 'process reward', 'prm', 'rlvr', 'grpo', 'deepseek-r1', 'r1', 'o1', 'test-time', 'search', 'math-shepherd', 'lean4']
    },
    {
        'id': 'track-5-programmatic-optimization',
        'title': 'Track 5: Programmatic Prompt Optimization & Compilers',
        'tagline': 'DSPy (MIPROv2), TextGrad text backpropagation, OPRO, and PromptBreeder evolutionary mutation.',
        'iconName': 'Zap',
        'color': 'from-indigo-500 to-violet-700',
        'keywords': ['dspy', 'textgrad', 'opro', 'promptbreeder', 'autopdl', 'promptwizard', 'compiler', 'teleprompter', 'evolutionary', 'optimization']
    },
    {
        'id': 'track-6-constrained-decoding',
        'title': 'Track 6: Constrained Decoding & Formal Grammar Engines',
        'tagline': 'CFG/EBNF pushdown automata, LLGuidance, XGrammar, subword token healing, and GSSD.',
        'iconName': 'ShieldCheck',
        'color': 'from-purple-500 to-pink-700',
        'keywords': ['grammar', 'constrained decoding', 'cfg', 'ebnf', 'json schema', 'llguidance', 'xgrammar', 'syncode', 'token healing', 'gssd', 'automata']
    },
    {
        'id': 'track-7-kv-cache-systems',
        'title': 'Track 7: Hardware-Aware KV Cache Systems, Paging & Latency',
        'tagline': 'PagedAttention, FlashAttention-3, MLA low-rank compression, SnapKV, KIVI 2-bit, and Mooncake.',
        'iconName': 'Server',
        'color': 'from-cyan-500 to-blue-700',
        'keywords': ['kv cache', 'pagedattention', 'flashattention', 'sarathi', 'mla', 'snapkv', 'kivi', 'mooncake', 'sequoia', 'specinfer', 'speculative decoding', 'roofline', 'compression']
    },
    {
        'id': 'track-8-mechanistic-interpretability',
        'title': 'Track 8: Mechanistic Interpretability & Circuit Analysis',
        'tagline': 'Sparse Autoencoders (JumpReLU), Gemma Scope, Transcoders, Causal Scrubbing, and ROME/MEMIT.',
        'iconName': 'Boxes',
        'color': 'from-fuchsia-500 to-rose-700',
        'keywords': ['sparse autoencoder', 'sae', 'jumprelu', 'gemma scope', 'transcoder', 'crosscoder', 'platonic', 'privileged basis', 'causal scrubbing', 'rome', 'memit']
    },
    {
        'id': 'track-9-alignment-mechanics',
        'title': 'Track 9: Representation Engineering & Alignment Mechanics',
        'tagline': 'Contrastive Activation Addition (CAA), Model Abliteration, PPO, DPO, KTO, SimPO, and Len-DPO.',
        'iconName': 'Sliders',
        'color': 'from-amber-500 to-orange-700',
        'keywords': ['activation addition', 'caa', 'repe', 'abliteration', 'dpo', 'kto', 'orpo', 'simpo', 'len-dpo', 'direct nash', 'alignment', 'prospect theory', 'meta-rewarding', 'self-improving']
    },
    {
        'id': 'track-10-agentic-and-multimodal',
        'title': 'Track 10: Autonomous Agents, Memory Topologies & Multimodal Streaming',
        'tagline': 'MemGPT OS paging, Zep temporal graphs, Reflexion, Magentic-One, Titans, and Project Astra.',
        'iconName': 'Boxes',
        'color': 'from-rose-500 to-red-700',
        'keywords': ['agent', 'memgpt', 'zep', 'reflexion', 'magentic-one', 'titans', 'astra', 'mimi', 'moshi', 'multimodal', 'codec', 'langgraph', 'swarm', 'actor model', 'vlm']
    },
    {
        'id': 'track-11-eval-and-safety',
        'title': 'Track 11: Benchmark Calibration, Evaluation & Frontier Red-Teaming',
        'tagline': 'Semantic Entropy epistemic uncertainty, G-Eval, MT-Bench, Many-Shot jailbreaks, and Crescendo.',
        'iconName': 'AlertTriangle',
        'color': 'from-emerald-600 to-lime-700',
        'keywords': ['semantic entropy', 'g-eval', 'mt-bench', 'jailbreak', 'crescendo', 'token smuggling', 'red-team', 'eval', 'calibration', 'arena', 'judge']
    },
    {
        'id': 'track-12-grand-monographs',
        'title': 'Track 12: Frontier Grand Technical Monographs & Treatises',
        'tagline': 'Exhaustive publication-grade treatises spanning TTC, Discrete Diffusion, Neurosymbolic Verification, Astra, and Memory.',
        'iconName': 'Award',
        'color': 'from-yellow-500 to-amber-700',
        'keywords': ['monograph']
    },
    {
        'id': 'track-13-micro-theories',
        'title': 'Track 13: Frontier Concept Snippets & Micro-Theories',
        'tagline': '370 atomic, mathematically-grounded micro-theories with Mermaid topology diagrams and formula proofs.',
        'iconName': 'Sparkles',
        'color': 'from-violet-500 to-fuchsia-700',
        'keywords': ['snippet']
    }
]

def clean_audio_script(title, body):
    text = body
    text = re.sub(r'```[\s\S]*?```', ' Code block omitted for audio narration. ', text)
    text = re.sub(r'\|[^\n]+\|', '', text)
    text = re.sub(r'\$\$[\s\S]*?\$\$', ' Mathematical formula omitted for narration. ', text)
    text = re.sub(r'\$[^\$\n]+\$', ' formula ', text)
    text = re.sub(r'###+\s*', '', text)
    text = re.sub(r'[*_#`]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    sentences = text.split('. ')
    selected = []
    curr_len = 0
    for s in sentences:
        if curr_len + len(s) < 550 and len(s) > 15:
            selected.append(s.strip())
            curr_len += len(s)
        if curr_len >= 350:
            break
    
    narration = f"Welcome to this lesson on {title}. " + ". ".join(selected) + "."
    return narration.replace('"', '\\"').replace('\n', ' ')

def extract_takeaways(body, default_title):
    takeaways = []
    lines = body.split('\n')
    for line in lines:
        if line.strip().startswith('- **') or line.strip().startswith('1. **') or line.strip().startswith('2. **') or line.strip().startswith('3. **'):
            clean = re.sub(r'[*_`]', '', line).strip()
            if len(clean) > 20:
                takeaways.append(clean[:140])
        if len(takeaways) >= 3:
            break
            
    if len(takeaways) < 2:
        takeaways = [
            f"Grounds the core architectural invariant of {default_title[:60]} in verified empirical literature.",
            "Eliminates heuristic trial-and-error by deriving exact mathematical formulations and systems trade-offs.",
            "Directly actionable for designing high-throughput, steerable frontier prompting workflows."
        ]
    return takeaways

def generate_quiz(title, body):
    return [
        {
            'id': 'q1',
            'question': f'What is the fundamental architectural advantage demonstrated in {title[:50]}?',
            'options': [
                'It reduces computational overhead while mathematically preserving or improving capability guarantees',
                'It requires retraining the entire model from scratch on CPU memory',
                'It enforces greedy temperature 0.0 on all decoding steps',
                'It disables multi-head attention in favor of RNNs'
            ],
            'correctIndex': 0,
            'explanation': f'{title[:60]} provides a rigorous, hardware- or mathematically-grounded optimization that avoids heuristic degradation.'
        },
        {
            'id': 'q2',
            'question': 'How does this mechanism translate into practical frontier prompt engineering?',
            'options': [
                'It allows prompt engineers to align input structure with the underlying latent activation dynamics of the model',
                'It replaces prompt engineering with random word generation',
                'It only applies to small 100M parameter models',
                'It prevents using system prompts altogether'
            ],
            'correctIndex': 0,
            'explanation': 'Understanding the model internal circuit mechanics allows practitioners to design prompts that activate optimal computational paths.'
        }
    ]

def make_clean_summary(body):
    text = body
    text = re.sub(r'^#+[^\n]*\n+', '', text)
    text = re.sub(r'^#+[^\n]*\n+', '', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'\|[^\n]+\|', '', text)
    text = re.sub(r'[*_`#]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:220].replace('"', '\\"')

def escape_content_for_ts(text):
    return text.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')

tracks_data = {t['id']: [] for t in TRACK_DEFINITIONS}

# =========================================================================
# 1. INGEST CURATED FOUNDATIONAL GUIDES (01_ to 04_) -> Track 1 (Primer)
# =========================================================================
print("1. Ingesting curated foundational guides into Track 1...")
curated_files = sorted(glob.glob(os.path.join(KEPLER_DIR, '0*/*.md')))
for c_file in curated_files:
    with open(c_file, 'r', encoding='utf-8') as f:
        body = f.read()
    
    # Title from first header
    title_match = re.search(r'^#\s+([^\n]+)', body, re.M)
    title = title_match.group(1).strip() if title_match else os.path.basename(c_file).replace('.md', '').replace('_', ' ').title()
    title = re.sub(r'^\d+\.\s*', '', title) # clean prefix numbers
    
    file_id = os.path.basename(c_file).replace('.md', '').replace('_', '-')
    lesson_id = f"primer-{file_id}"
    
    lesson_obj = {
        'id': lesson_id,
        'title': f"Primer: {title}",
        'estimatedMinutes': max(8, min(20, len(body) // 600)),
        'difficulty': 'Intermediate' if '01_' in c_file else ('Advanced' if '02_' in c_file or '03_' in c_file else 'Frontier'),
        'summary': make_clean_summary(body),
        'keyTakeaways': extract_takeaways(body, title),
        'audioScript': clean_audio_script(title, body),
        'content': escape_content_for_ts(body),
        'discussionPrompts': [
            f"How does mastering {title[:40]} prevent non-deterministic failures in production LLM applications?",
            "What architectural trade-offs exist between programmatic prompt generation vs manual human prompt optimization?"
        ],
        'quiz': generate_quiz(title, body)
    }
    tracks_data['track-1-primer'].append(lesson_obj)

# =========================================================================
# 2. INGEST STANDALONE RAW RESEARCH REPORTS -> Tracks 2 to 11
# =========================================================================
print("2. Ingesting standalone deep research reports...")
raw_reports = sorted([f for f in glob.glob(os.path.join(KEPLER_DIR, 'obsidian_vault', 'raw_research', '*.md')) 
                      if 'monograph' not in os.path.basename(f) and 'dossier' not in os.path.basename(f)])

for r_file in raw_reports:
    with open(r_file, 'r', encoding='utf-8') as f:
        body = f.read()
    
    title_match = re.search(r'^#\s+([^\n]+)', body, re.M)
    title = title_match.group(1).strip() if title_match else os.path.basename(r_file).replace('.md', '').replace('_', ' ').title()
    file_id = os.path.basename(r_file).replace('.md', '').replace('_', '-')
    
    # Route by keywords
    matched_track_id = 'track-2-icl-physics'
    best_score = 0
    search_text = (title + " " + body[:400]).lower()
    for t in TRACK_DEFINITIONS[1:11]: # Tracks 2 to 11
        score = sum(1 for kw in t.get('keywords', []) if kw in search_text)
        if score > best_score:
            best_score = score
            matched_track_id = t['id']
            
    lesson_obj = {
        'id': f"report-{file_id}",
        'title': f"Deep Dive: {title}",
        'estimatedMinutes': max(10, min(25, len(body) // 600)),
        'difficulty': 'Frontier',
        'summary': make_clean_summary(body),
        'keyTakeaways': extract_takeaways(body, title),
        'audioScript': clean_audio_script(title, body),
        'content': escape_content_for_ts(body),
        'discussionPrompts': [
            f"In what scenarios would deploying {title[:40]} introduce latency or compute bottlenecks?",
            "How does this breakthrough reshape current prompt architecture paradigms?"
        ],
        'quiz': generate_quiz(title, body)
    }
    tracks_data[matched_track_id].append(lesson_obj)

# =========================================================================
# 3. INGEST LATENT MECHANICS DOSSIER (329 SECTIONS) -> Tracks 2 to 11
# =========================================================================
print("3. Ingesting Latent Mechanics Dossier (329 sections)...")
dossier_path = os.path.join(KEPLER_DIR, 'obsidian_vault', 'raw_research', 'latent_mechanics_dossier.md')
with open(dossier_path, 'r', encoding='utf-8') as f:
    dossier_text = f.read()

section_matches = list(re.finditer(r'##\s+(\d+)\.\s+([^\n]+)', dossier_text))
for i, m in enumerate(section_matches):
    sec_num = int(m.group(1))
    sec_title = m.group(2).strip()
    start_pos = m.end()
    end_pos = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(dossier_text)
    sec_body = dossier_text[start_pos:end_pos].strip()
    
    matched_track_id = 'track-2-icl-physics'
    best_score = 0
    search_text = (sec_title + " " + sec_body[:300]).lower()
    for t in TRACK_DEFINITIONS[1:11]: # Tracks 2 to 11
        score = sum(1 for kw in t.get('keywords', []) if kw in search_text)
        if score > best_score:
            best_score = score
            matched_track_id = t['id']
            
    clean_slug = re.sub(r'[^a-zA-Z0-9]+', '-', sec_title.lower())[:30]
    lesson_id = f"lesson-sec-{sec_num}-{clean_slug}"
    
    lesson_obj = {
        'id': lesson_id,
        'title': f"§{sec_num}: {sec_title}",
        'estimatedMinutes': max(5, min(15, len(sec_body) // 800)),
        'difficulty': 'Frontier' if sec_num > 250 else ('Advanced' if sec_num > 100 else 'Intermediate'),
        'summary': make_clean_summary(sec_body),
        'keyTakeaways': extract_takeaways(sec_body, sec_title),
        'audioScript': clean_audio_script(sec_title, sec_body),
        'content': escape_content_for_ts(sec_body),
        'discussionPrompts': [
            f"How does {sec_title[:40]} fundamentally shift our understanding of frontier LLM steerability?",
            "What failure modes emerge if this mechanism is deployed without proper calibration?"
        ],
        'quiz': generate_quiz(sec_title, sec_body)
    }
    tracks_data[matched_track_id].append(lesson_obj)

# =========================================================================
# 4. INGEST STANDALONE MONOGRAPHS -> Track 12 (Grand Monographs)
# =========================================================================
print("4. Ingesting Standalone Grand Monographs...")
monograph_files = sorted(glob.glob(os.path.join(KEPLER_DIR, 'obsidian_vault', 'raw_research', '*monograph*.md')))
for m_idx, m_path in enumerate(monograph_files):
    with open(m_path, 'r', encoding='utf-8') as f:
        m_body = f.read()
        
    m_title_match = re.search(r'^#\s+([^\n]+)', m_body, re.M)
    m_title = m_title_match.group(1).strip() if m_title_match else os.path.basename(m_path).replace('.md', '').replace('_', ' ').title()
    clean_slug = re.sub(r'[^a-zA-Z0-9]+', '-', m_title.lower())[:30]
    
    lesson_obj = {
        'id': f"treatise-{m_idx + 1}-{clean_slug}",
        'title': f"Treatise {m_idx + 1}: {m_title}",
        'estimatedMinutes': max(20, min(45, len(m_body) // 700)),
        'difficulty': 'Frontier',
        'summary': make_clean_summary(m_body),
        'keyTakeaways': extract_takeaways(m_body, m_title),
        'audioScript': clean_audio_script(m_title, m_body[:3000]),
        'content': escape_content_for_ts(m_body),
        'discussionPrompts': [
            f"What makes the findings in '{m_title[:40]}' foundational for the next decade of frontier AI?",
            "How can high-reliability production systems implement these theoretical principles without prohibitive latency?"
        ],
        'quiz': generate_quiz(m_title, m_body)
    }
    tracks_data['track-12-grand-monographs'].append(lesson_obj)

# =========================================================================
# 5. INGEST ATOMIC CONCEPT SNIPPETS (370 FILES) -> Track 13 (Micro-Theories)
# =========================================================================
print("5. Ingesting 370 Atomic Concept Snippets into Track 13...")
snippet_files = sorted(glob.glob(os.path.join(KEPLER_DIR, 'obsidian_vault', 'raw_research', 'snippets', '*.md')))
for s_idx, s_path in enumerate(snippet_files):
    with open(s_path, 'r', encoding='utf-8') as f:
        s_body = f.read()
        
    s_title_match = re.search(r'^#\s+([^\n]+)', s_body, re.M)
    s_title = s_title_match.group(1).strip() if s_title_match else os.path.basename(s_path).replace('.md', '').replace('_', ' ').title()
    clean_slug = os.path.basename(s_path).replace('.md', '').replace('_', '-')
    
    lesson_obj = {
        'id': f"snippet-{s_idx + 1}-{clean_slug[:30]}",
        'title': f"Snippet {s_idx + 1}: {s_title}",
        'estimatedMinutes': max(3, min(8, len(s_body) // 400)),
        'difficulty': 'Advanced',
        'summary': make_clean_summary(s_body),
        'keyTakeaways': extract_takeaways(s_body, s_title),
        'audioScript': clean_audio_script(s_title, s_body),
        'content': escape_content_for_ts(s_body),
        'discussionPrompts': [
            f"How does {s_title[:40]} relate to broader latent activation steering?",
            "In what real-world prompt scenarios does this specific concept provide an unfair advantage?"
        ],
        'quiz': generate_quiz(s_title, s_body)
    }
    tracks_data['track-13-micro-theories'].append(lesson_obj)

# =========================================================================
# ASSEMBLE MASTER CURRICULUM AND WRITE TO TS FILE
# =========================================================================
print("\n--- Summary of Assembled Tracks ---")
master_tracks = []
total_modules = 0

for t_def in TRACK_DEFINITIONS:
    lessons = tracks_data[t_def['id']]
    total_modules += len(lessons)
    print(f"- {t_def['title']}: {len(lessons)} modules")
    
    track_dict = {
        'id': t_def['id'],
        'title': t_def['title'],
        'tagline': t_def['tagline'],
        'iconName': t_def['iconName'],
        'color': t_def['color'],
        'lessons': lessons
    }
    master_tracks.append(track_dict)

print(f"\nTOTAL ALL-INCLUSIVE MODULES IN CURRICULUM: {total_modules}")

ts_content = """import { CourseTrack } from '../types';

/**
 * THE COMPLETE ALL-INCLUSIVE FRONTIER PROMPT MASTERY CURRICULUM
 * 
 * Auto-compiled from prompt-engineering-mastery knowledge base:
 * - Master Foundations & Core Principles Primer (01_ to 04_)
 * - Standalone Deep Research Reports
 * - Latent Mechanics Dossier (§1 to §329)
 * - Frontier Grand Technical Monographs & Treatises (Treatises 1 to 8)
 * - Frontier Concept Snippets & Micro-Theories (Snippets 1 to 370)
 * 
 * Total Master Tracks: 13
 * Total Fully-Interactive Offline Modules: """ + str(total_modules) + """
 */

export const COURSE_CURRICULUM: CourseTrack[] = """

# Convert to JSON with template literals for markdown content
print("Writing curriculum.ts with escaped template strings...")
with open(OUTPUT_CURRICULUM_PATH, 'w', encoding='utf-8') as out_f:
    out_f.write(ts_content)
    out_f.write("[\n")
    
    for t_idx, track in enumerate(master_tracks):
        out_f.write("  {\n")
        out_f.write(f'    id: "{track["id"]}",\n')
        out_f.write(f'    title: {json.dumps(track["title"])},\n')
        out_f.write(f'    tagline: {json.dumps(track["tagline"])},\n')
        out_f.write(f'    iconName: "{track["iconName"]}",\n')
        out_f.write(f'    color: "{track["color"]}",\n')
        out_f.write("    lessons: [\n")
        
        for l_idx, l in enumerate(track['lessons']):
            out_f.write("      {\n")
            out_f.write(f'        id: "{l["id"]}",\n')
            out_f.write(f'        title: {json.dumps(l["title"])},\n')
            out_f.write(f'        estimatedMinutes: {l["estimatedMinutes"]},\n')
            out_f.write(f'        difficulty: "{l["difficulty"]}",\n')
            out_f.write(f'        summary: {json.dumps(l["summary"])},\n')
            out_f.write(f'        keyTakeaways: {json.dumps(l["keyTakeaways"])},\n')
            out_f.write(f'        audioScript: {json.dumps(l["audioScript"])},\n')
            out_f.write(f'        content: `{l["content"]}`,\n')
            out_f.write(f'        discussionPrompts: {json.dumps(l["discussionPrompts"])},\n')
            out_f.write(f'        quiz: {json.dumps(l["quiz"])}\n')
            out_f.write("      }" + ("," if l_idx + 1 < len(track['lessons']) else "") + "\n")
            
        out_f.write("    ]\n")
        out_f.write("  }" + ("," if t_idx + 1 < len(master_tracks) else "") + "\n")
        
    out_f.write("];\n")

print(f"Successfully compiled all {total_modules} modules to {OUTPUT_CURRICULUM_PATH}!")
