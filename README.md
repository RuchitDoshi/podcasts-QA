# Podcasts Q&A — RAG + Agentic System

A Staff-level portfolio project: a retrieval-augmented, tool-calling agent that answers
questions about the Lex Fridman Podcast, grounded strictly in the actual transcript
content — with every claim in every answer traceable to a real, verified chunk of audio.

This README is the short version. The full architecture doc (18 sections, investigation
write-ups, and every real bug found and fixed along the way) lives at
`podcast_qa_architecture.docx`.

## What this is

- **Ingestion**: RSS → download → Whisper transcription → speaker diarization → sponsor-aware
  chunking → hybrid dense+BM25 indexing, across 30 real episodes (~7,300 chunks)
- **Retrieval**: hybrid search (dense embeddings + BM25), a cross-encoder reranker, and an
  empirically-tuned confidence threshold that lets the system honestly say "I don't know"
  instead of guessing
- **Agent**: a hand-rolled tool-calling loop (search transcripts, look up episodes, expand
  context, submit a cited answer) with citation verification that mechanically checks every
  claim was actually shown to the model — not just trusted
- **Eval, not vibes**: retrieval metrics (hit@1, MRR, precision@k), agent trajectory metrics
  (task success, tool selection accuracy, error recovery, cost/latency), and answer-quality
  metrics (faithfulness, relevancy, correctness) — all measured, not assumed

## Headline findings

**Ad content pollution, found and fixed.** An adversarial query ("what companies has Lex
mentioned recently") originally returned **100% ad content** in its top results. Sponsor-name
extraction + split-phrase detection + a chunk-boundary bug fix brought that down to **0% on
real queries**, confirmed on real re-processed data.

**Retrieval quality (post-reranker, current corpus):**

| Metric | Value |
|---|---|
| hit@1 | 100% (n=8) |
| MRR | 1.000 |
| Avg precision@k | 84.4% |
| Retrieval null rate | 0.0% |

**Agent trajectory eval (27 scenarios, `openai/gpt-oss-120b`):**

| Metric | Value |
|---|---|
| Task success rate | 84.2% (n=19) |
| Tool selection accuracy | 95.5% |
| Trajectory efficiency | 0.93 |

Three real correctness bugs were found and fixed by measuring this rather than trusting it —
including a case where a structurally-correct answer was being discarded because the citation
schema had no valid shape for episode-metadata questions. See the architecture doc, Section 18,
for the full investigation.

**Answer-quality comparison across three models, same corpus, same eval set:**

| Metric | `gpt-oss-120b` | `MiniMax-M2.5` | `Claude Sonnet 4.5` |
|---|---|---|---|
| Faithfulness | 0.804 | 0.920 | **0.988** |
| Relevancy | 0.808 | **0.822** | 0.806 |
| Correctness (vs. reference) | 0.500 | 0.587 | **0.761** |
| Declined (of 12) | 4 | 3 | **2** |
| Avg cost/query | **$0.0015** | $0.0040 (2.7x) | $0.0584 (39x) |
| Avg latency | **12.0s** | 24.4s | 28.5s |

Faithfulness and correctness both scale cleanly with model tier; relevancy stays roughly flat
across all three — these models are all comparably good at staying on-topic, but differ
substantially in avoiding unsupported claims and matching a complete, correct answer. Sonnet's
quality gain is real, but so is a **39x cost multiple** over `gpt-oss-120b` — the right choice
genuinely depends on the use case (high-volume/cost-sensitive vs. low-volume/high-stakes), not
a universal "best" model.

*(Groq/Llama was the original default provider for this project and is used throughout most of
the architecture doc's investigations, but was dropped from this specific comparison due to
persistent free-tier daily quota limits that made repeated evaluation runs impractical.)*

## Tech stack

Python · Whisper (transcription) · pyannote.audio (diarization) · `bge-small` (embeddings) ·
ChromaDB (vector store) · BM25 (`rank_bm25`) · a cross-encoder reranker
(`ms-marco-MiniLM-L-6-v2`) · LangChain + LiteLLM (multi-provider LLM access) · Groq, and a
company LiteLLM proxy exposing gpt-oss, MiniMax, and Claude · Gemini/LiteLLM as an
independent LLM-judge for eval

## Running it

```bash
pip install -r requirements.txt

# Ingest (once)
python data/pull_rss_episodes.py && python data/download.py && python data/transcribe.py
python ingest/diarize.py && python ingest/sponsor_extractor.py && python ingest/chunk.py
python ingest/index.py

# Ask a question
python agent/orchestrator.py --query "What does Kaldellis say about Justinian's legal reforms?" --verbose

# Run the evals
python eval/eval_retrieval.py
python eval/eval_agent.py
python eval/generate_references.py && python eval/eval_answers.py
```

Add `--provider litellm --model "<model-id>"` to any agent/eval command to run against a
different LLM provider (see `agent/orchestrator.py`'s `PROVIDER_DEFAULT_MODELS`).

## Known, deliberately deferred gaps

Named honestly rather than hidden: recall@k/nDCG@k (needs chunk-level relevance labels, a real
labeling investment not yet made), MinHash near-duplicate detection, speaker name mapping
(`SPEAKER_00` → real names — built, tested, deprioritized as display polish), multi-turn
conversation memory, and a runtime self-critique guardrail (distinct from the offline eval
built here). Full reasoning for each in the architecture doc.

## Full documentation

`podcast_qa_architecture.docx` — system design, every investigation (ad detection, retrieval
tuning, reranker A/B, the LiteLLM/LangChain integration journey, the trajectory-eval bug hunt),
and the reasoning behind every deferred decision.
