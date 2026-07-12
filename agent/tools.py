"""
Tools available to the agent orchestrator. Each tool is a plain Python
function plus a JSON schema describing it for the LLM's tool-calling API
(Groq's chat completions endpoint is OpenAI-compatible, so this uses the
standard OpenAI function-calling schema shape).

Four tools, matching the routing behavior described in the architecture
doc: search within/across episodes, look up which episode(s) match a guest
or topic (so the agent can resolve "the Kaldellis episode" to an
episode_id before searching), pull an episode's own metadata for
summary-style questions that don't need a chunk search at all, and expand
context around a specific chunk when the retrieved text alone doesn't
fully answer the question.

Neighbor-chunk design: search_transcripts does NOT auto-expand results with
surrounding context by default -- that would roughly double retrieval-stage
token cost on the common case where the top chunk is already sufficient
(true for most of this project's eval queries). Instead, each search result
carries lightweight boundary metadata (neighboring chunk_ids and their
timestamps, no text) so the model can SEE that more context exists nearby,
then explicitly call get_context_window only when it judges the retrieved
text incomplete. This keeps the pattern consistent with everything else in
this file: the LLM decides when to spend the extra tool call, nothing is
silently injected.

This module holds no state of its own -- the corpus/bm25/embedder/collection
are loaded once by the orchestrator and passed in, so tools.py stays a pure
function library that's easy to unit test in isolation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieve.hybrid import hybrid_search  # noqa: E402


def build_episode_index(corpus: list[dict]) -> dict[str, list[str]]:
    """Groups chunk_ids by episode_id, ordered by start time -- the ordering
    that makes "neighboring chunk" a well-defined concept. Built once by the
    caller (AgentContext) and passed down, not rebuilt per tool call, since
    it only depends on the corpus, not the query."""
    by_episode: dict[str, list[dict]] = {}
    for record in corpus:
        by_episode.setdefault(record["episode_id"], []).append(record)

    return {
        episode_id: [r["chunk_id"] for r in sorted(records, key=lambda r: r["start"])]
        for episode_id, records in by_episode.items()
    }


def get_neighbor_ids(
    chunk_id: str, episode_id: str, episode_index: dict[str, list[str]]
) -> tuple[str | None, str | None]:
    """Returns (prev_chunk_id, next_chunk_id) within the same episode, or
    None at either end of the episode's chunk sequence."""
    order = episode_index.get(episode_id)
    if not order or chunk_id not in order:
        return None, None

    idx = order.index(chunk_id)
    prev_id = order[idx - 1] if idx > 0 else None
    next_id = order[idx + 1] if idx < len(order) - 1 else None
    return prev_id, next_id


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_transcripts",
            "description": (
                "Search podcast transcript chunks by meaning and keywords. Use this "
                "to answer any question about what was discussed, said, or explained "
                "in one or more episodes. If the question is about a specific guest "
                "or episode, pass episode_id to restrict the search to that episode "
                "-- use list_episodes first if you don't already know the episode_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query -- a natural-language question or topic.",
                    },
                    "episode_id": {
                        "type": "string",
                        "description": "Restrict search to this episode only. Omit to search across all episodes.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return. Default 5.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_episodes",
            "description": (
                "Look up which episode(s) match a guest name or topic tag. Use this "
                "first when the user refers to a guest or topic by name and you need "
                "the episode_id before calling search_transcripts, or when the user "
                "asks a cross-episode question like 'which episodes discuss X'. "
                "Provide at least one of guest or tag."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "guest": {
                        "type": "string",
                        "description": "Guest name or partial name to match, case-insensitive.",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Topic tag to match, e.g. 'history', 'ai', 'physics'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_episode_info",
            "description": (
                "Get an episode's own metadata (title, guest, date, tags, description). "
                "Use this for questions about the episode itself (who was the guest, "
                "when did it air, what is it about) rather than its content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "episode_id": {"type": "string", "description": "The episode id."},
                },
                "required": ["episode_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_context_window",
            "description": (
                "Get the text of chunks immediately before and/or after a specific chunk, "
                "within the same episode. Use this when a search_transcripts result seems "
                "incomplete or cut off -- e.g. it references something not fully explained, "
                "or ends mid-thought -- and you need the surrounding conversation to answer "
                "accurately. Each search_transcripts result includes prev_chunk_id and "
                "next_chunk_id fields (null at an episode's start/end) -- pass one of those "
                "as chunk_id here to expand around it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk_id to expand context around (from a prior search_transcripts result).",
                    },
                    "before": {
                        "type": "integer",
                        "description": "How many preceding chunks to include. Default 1.",
                    },
                    "after": {
                        "type": "integer",
                        "description": "How many following chunks to include. Default 1.",
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
]


def search_transcripts_tool(
    query: str,
    corpus,
    corpus_by_id,
    bm25,
    collection,
    embedder,
    settings,
    episode_id: str | None = None,
    top_k: int | None = None,
    exclude_ads: bool = True,
    cross_encoder=None,
    episode_index: dict[str, list[str]] | None = None,
) -> dict:
    """Runs hybrid search, optionally restricted to a single episode.

    Restriction is applied post-hoc on the merged candidate pool rather than
    pre-filtering the dense/BM25 search itself -- simpler, and cheap at this
    corpus size (a few thousand chunks). If precision within a single long
    episode ever becomes a bottleneck, this is the place to push the filter
    down into the ChromaDB query's `where` clause instead.

    Confidence gating: results below retrieval.min_score_threshold (in
    settings.yaml) are dropped before being returned to the model, and if
    NOTHING clears the bar, the tool returns an explicit no-match signal
    instead of quietly handing back weak matches dressed up as answers --
    the model is instructed to treat that as grounds to say "I don't know"
    rather than synthesize from low-relevance chunks. This threshold is a
    blunt instrument; tune it against eval/eval_retrieval.py results rather
    than guessing, since too high a bar will start rejecting real answers.

    Confidence gating: results are checked against retrieval.min_score_threshold
    (settings.yaml) applied to raw_dense_similarity specifically -- NOT the
    fused hybrid score. The fused score is min-max normalized across the
    candidate pool, which always stretches the best-of-the-pool result
    toward 1.0 even when every candidate is a poor match, so it can never
    reliably signal "nothing here is relevant." Raw cosine similarity is
    the one number in this pipeline that means something in an absolute
    sense, so it's the only one worth thresholding for a no-answer signal.
    Tune the threshold against eval/eval_retrieval.py results, not a guess.

    Returns chunk_id and score alongside each result (not just text/episode/
    timestamp) -- the orchestrator needs chunk_id for session-level dedup
    and score for logging, even though the model itself is never shown
    those two fields directly.
    """
    settings = dict(settings)
    settings["retrieval"] = dict(settings["retrieval"])
    if top_k:
        settings["retrieval"]["top_k"] = top_k * 3 if episode_id else top_k

    results = hybrid_search(
        query,
        corpus,
        corpus_by_id,
        bm25,
        collection,
        embedder,
        settings,
        exclude_ads=exclude_ads,
        cross_encoder=cross_encoder,
    )

    if episode_id:
        results = [r for r in results if r["episode_id"] == episode_id]

    final_k = top_k or settings["retrieval"]["top_k"]
    results = results[:final_k]

    min_similarity = settings["retrieval"].get("min_score_threshold", 0.0)
    passing = [r for r in results if r["raw_dense_similarity"] >= min_similarity]

    if not passing:
        return {
            "results": [],
            "no_relevant_results": True,
            "note": (
                "No chunks met the minimum relevance threshold for this query. "
                "Treat this as evidence the corpus does not contain a good answer -- "
                "do not guess or answer from general knowledge."
            ),
        }

    result_items = []
    for r in passing:
        prev_id, next_id = (None, None)
        if episode_index:
            prev_id, next_id = get_neighbor_ids(r["chunk_id"], r["episode_id"], episode_index)

        result_items.append(
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "episode_id": r["episode_id"],
                "guest": r["metadata"].get("guest"),
                "start": r["start"],
                "end": r["end"],
                "score": round(r["score"], 3),
                # Boundary metadata only -- no text -- so the model can SEE more
                # context exists without paying the token cost unless it decides
                # to actually call get_context_window. prev_end/next_start let it
                # judge how much of a time gap expanding would actually cover.
                "prev_chunk_id": prev_id,
                "prev_end": corpus_by_id[prev_id]["end"] if prev_id else None,
                "next_chunk_id": next_id,
                "next_start": corpus_by_id[next_id]["start"] if next_id else None,
            }
        )

    return {"results": result_items, "no_relevant_results": False}


def list_episodes_tool(
    episode_meta: dict[str, dict], guest: str | None = None, tag: str | None = None
) -> list[dict]:
    matches = []
    for ep_id, meta in episode_meta.items():
        if guest and guest.lower() not in (meta.get("guest") or "").lower():
            continue
        if tag and tag.lower() not in [t.lower() for t in (meta.get("tags") or [])]:
            continue
        matches.append(
            {
                "episode_id": ep_id,
                "title": meta.get("title"),
                "guest": meta.get("guest"),
                "date": meta.get("date"),
                "tags": meta.get("tags"),
            }
        )
    return matches


def get_episode_info_tool(episode_meta: dict[str, dict], episode_id: str) -> dict:
    meta = episode_meta.get(episode_id)
    if not meta:
        return {"error": f"No episode found with id {episode_id!r}"}
    return {
        "episode_id": episode_id,
        "title": meta.get("title"),
        "guest": meta.get("guest"),
        "date": meta.get("date"),
        "tags": meta.get("tags"),
        "description": meta.get("description"),
    }


def get_context_window_tool(
    chunk_id: str,
    corpus_by_id: dict[str, dict],
    episode_index: dict[str, list[str]],
    before: int = 1,
    after: int = 1,
) -> dict:
    """Expands context around a specific chunk by walking `before`/`after`
    steps through the episode's chunk order. Returns each included chunk
    (center + neighbors) with its own id/timestamps/text -- not merged into
    one blob -- so the model can still cite the specific chunk a claim
    actually came from, same as any search_transcripts result."""
    center = corpus_by_id.get(chunk_id)
    if not center:
        return {
            "error": f"chunk_id {chunk_id!r} not found -- it must come from a prior search_transcripts result."
        }

    episode_id = center["episode_id"]
    order = episode_index.get(episode_id, [])
    if chunk_id not in order:
        return {
            "error": f"chunk_id {chunk_id!r} not found in episode {episode_id!r}'s chunk order."
        }

    idx = order.index(chunk_id)
    start_idx = max(0, idx - max(before, 0))
    end_idx = min(len(order), idx + max(after, 0) + 1)
    window_ids = order[start_idx:end_idx]

    chunks = []
    for cid in window_ids:
        record = corpus_by_id[cid]
        chunks.append(
            {
                "chunk_id": cid,
                "text": record["text"],
                "start": record["start"],
                "end": record["end"],
                "is_center": cid == chunk_id,
            }
        )

    return {"episode_id": episode_id, "chunks": chunks}
