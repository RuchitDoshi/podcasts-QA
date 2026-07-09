# Podcasts Q&A — RAG + Agentic system

## Project
Retrieval-augmented Q&A over long-form spoken content (podcasts, lectures). Whisper
transcription -> audio-aware chunking -> hybrid retrieval -> agentic answer layer ->
eval harness. Target: Staff ML Engineer portfolio project #3 (RAG/Agentic track).

Display name: "Podcasts Q&A". Repo/folder name: `podcasts_qa` (no `&` — shell-unsafe).

## Stack
- Python 3.14.3 (pyenv local, isolated per-project venv)
- faster-whisper (self-hosted ASR, CPU or Colab T4)
- sentence-transformers (bge-small or e5-base, CPU embeddings)
- chromadb (local vector store)
- rank-bm25 (keyword/sparse retrieval side of hybrid search)
- langgraph (agent orchestration) — hand-rolled fallback if this feels heavier than needed
- groq (OpenAI-compatible client) for the answer LLM — Llama 3.3 70B, free tier
- google-generativeai for the judge LLM — Gemini 2.5 Flash, free tier, separate model
  family from the answerer (avoids self-preference bias in eval)
- yt-dlp for audio acquisition
- ragas (or a hand-rolled harness) for retrieval + answer eval

## Directory structure
podcasts_qa/
├── CLAUDE.md
├── requirements.txt
├── config/
│   ├── episodes.yaml         # list of source episodes (RSS/YouTube URLs + metadata)
│   └── settings.yaml         # model names, chunk size, retrieval weights
├── data/
│   ├── download.py           # pulls raw audio via yt-dlp, writes data/raw/
│   ├── transcribe.py         # faster-whisper transcription -> data/transcripts/
│   ├── pull_rss_episodes.py  # builds episodes.yaml from the podcast RSS feed
│   ├── enrich_episodes.py    # fixes guest names + generates tags via LLM
│   ├── raw/                  # audio files (gitignored)
│   └── transcripts/          # JSON transcripts with word-level timestamps (gitignored)
├── notebooks/
│   └── podcasts_qa_transcribe.ipynb   # GPU (T4) transcription run on Colab free tier
├── ingest/
│   ├── chunk.py               # speaker-turn / timestamp-aware chunking
│   └── index.py                # embeddings + chromadb indexing
├── retrieve/
│   └── hybrid.py               # dense + BM25 hybrid retrieval
├── agent/
│   ├── orchestrator.py          # single-episode vs cross-episode routing, tool calls
│   └── tools.py                 # jump_to_timestamp, summarize_episode, compare_episodes
├── eval/
│   ├── eval_retrieval.py        # retrieval null rate, precision/recall
│   ├── eval_answers.py          # LLM-as-judge scoring (Gemini)
│   └── adversarial_set.json     # ambiguous / no-answer / cross-episode eval questions
├── serve/
│   └── app.py                   # FastAPI entrypoint for demo
└── tests/
    └── test_chunking.py

## Conventions
- All audio resampled to 16kHz mono before transcription
- Transcript JSON schema: {"episode_id": ..., "segments": [{"start": float, "end": float,
  "speaker": str|null, "text": str}]}
- Chunk schema adds: {"chunk_id": ..., "episode_id": ..., "start": float, "end": float,
  "text": str, "metadata": {...}}
- Never commit audio files, transcripts, or vector store data — see .gitignore
- API keys (GROQ_API_KEY, GOOGLE_API_KEY) loaded from .env, never hardcoded
- Config-driven: episode list and model/chunking parameters live in config/*.yaml, not
  hardcoded in scripts
- Judge LLM must always be a different model family from the answer LLM (bias mitigation)

## Commands
- /pull-episodes: python data/pull_rss_episodes.py --limit 30
- /enrich-episodes: python data/enrich_episodes.py --config config/episodes.yaml
- /download: python data/download.py --config config/episodes.yaml
- /transcribe: python data/transcribe.py --input data/raw --output data/transcripts
- /chunk: python ingest/chunk.py --input data/transcripts --output data/chunks
- /index: python ingest/index.py --input data/chunks
- /serve: uvicorn serve.app:app --reload
- /eval-retrieval: python eval/eval_retrieval.py
- /eval-answers: python eval/eval_answers.py

## Claude Code notes
- When debugging slow transcription: check if faster-whisper is running int8 quantization
  (CPU) vs float16 (GPU) — int8 is the right default for local CPU runs
- When adding a new episode source: update config/episodes.yaml and re-run /download,
  /transcribe, /chunk, /index in sequence — don't skip steps
- Chunking must stay timestamp-anchored — never fall back to naive fixed-size chunking,
  it's the whole point of the audio-aware differentiation
- Whisper transcripts should be streamed/batched per-episode, never all loaded into memory
  at once
- Keep answer LLM (Groq/Llama) and judge LLM (Gemini) on separate provider calls —
  never let the same model judge its own output
