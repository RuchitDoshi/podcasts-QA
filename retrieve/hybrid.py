"""
Hybrid retrieval: combines dense (embedding/ChromaDB) search with BM25
(keyword) search over the same chunk corpus, merged by a weighted score.

Why hybrid, not dense-only: dense retrieval is strong on semantic/topical
match (as the sanity check showed) but can miss exact matches on named
entities, jargon, or unusual proper nouns that a query shares verbatim with
a chunk -- BM25 is the classic fix, since it scores on literal term overlap
rather than embedding proximity.

BM25 index is built in-memory from data/chunks/*.json at startup -- the
corpus size here (a few thousand chunks) makes this cheap enough not to need
a persisted sparse index, unlike the dense side which lives in ChromaDB.

This module can be used as a library (import hybrid_search) by the agent
layer, or run directly for ad-hoc query testing from the command line.

Usage:
    python retrieve/hybrid.py --query "What does Kaldellis say about the fall of Constantinople?"
    python retrieve/hybrid.py --query "..." --top-k 10 --exclude-ads
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hybrid")

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_settings(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_corpus(chunks_dir: Path) -> list[dict]:
    """Load all chunk records across all episodes into one flat list --
    this is the shared corpus both dense and BM25 search over."""
    corpus = []
    for chunk_file in sorted(chunks_dir.glob("*.json")):
        with open(chunk_file) as f:
            corpus.extend(json.load(f))
    return corpus


def build_bm25(corpus: list[dict]):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.error("rank_bm25 not installed. Run: pip install rank_bm25")
        sys.exit(1)

    tokenized = [tokenize(r["text"]) for r in corpus]
    return BM25Okapi(tokenized)


def get_embedder(settings: dict):
    from sentence_transformers import SentenceTransformer

    e = settings["embeddings"]
    return SentenceTransformer(e["model_name"], device=e["device"])


def get_collection(settings: dict):
    import chromadb

    client = chromadb.PersistentClient(path=settings["paths"]["chroma_dir"])
    return client.get_collection("podcast_chunks")


def minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def dense_search(
    collection, embedder, query: str, n_results: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Returns ({chunk_id: normalized_score}, {chunk_id: raw_cosine_similarity}).

    The raw similarity is kept separate from the normalized one on purpose:
    min-max normalization is relative to the candidate pool, so it always
    stretches the best-of-the-pool result toward 1.0 even when every
    candidate is a poor match -- it cannot be used to detect "nothing here
    is actually relevant." Raw cosine similarity is the only one of the two
    that carries meaning in an absolute sense, so it's what confidence
    gating (agent/tools.py) should threshold against, not the fused score.
    """
    query_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=n_results)

    ids = results["ids"][0]
    distances = results["distances"][0]  # cosine distance, lower is better

    # convert distance -> similarity, then normalize so higher is always better
    similarities = [1 - d for d in distances]
    normalized = minmax_normalize(similarities)
    raw = dict(zip(ids, similarities, strict=True))
    return dict(zip(ids, normalized, strict=True)), raw


def bm25_search(bm25, corpus: list[dict], query: str, n_results: int) -> dict[str, float]:
    """Returns {chunk_id: normalized_score}, higher is better."""
    scores = bm25.get_scores(tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

    top_scores = [scores[i] for i in ranked_idx]
    normalized = minmax_normalize(top_scores)

    return {corpus[i]["chunk_id"]: norm for i, norm in zip(ranked_idx, normalized, strict=True)}


def get_cross_encoder(settings: dict):
    """Loads the cross-encoder reranker model once, for reuse across calls --
    same lazy-load-once-and-pass-down pattern as get_embedder. Runs locally
    on CPU, no API key or rate limit, unlike the Groq/Gemini calls elsewhere
    in this project."""
    from sentence_transformers import CrossEncoder

    r = settings["retrieval"]
    model_name = r.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    device = r.get("reranker_device", "cpu")
    return CrossEncoder(model_name, device=device)


def rerank_pool(query: str, candidates: list[dict], cross_encoder) -> list[dict]:
    """Re-scores and re-sorts a candidate pool using a cross-encoder, which
    jointly encodes (query, passage) pairs rather than comparing independent
    embeddings -- generally more accurate than bi-encoder + BM25 fusion alone
    for final ranking, at the cost of one forward pass per candidate instead
    of a single batched embedding call. Adds a "rerank_score" field and
    re-sorts by it; the original fused "score" is kept on each record for
    comparison/debugging, not discarded."""
    if not candidates:
        return candidates

    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = cross_encoder.predict(pairs)

    for candidate, score in zip(candidates, rerank_scores, strict=True):
        candidate["rerank_score"] = float(score)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


def rrf_fusion(
    dense_scores: dict[str, float], sparse_scores: dict[str, float], k: int = 60
) -> dict[str, float]:
    """Reciprocal Rank Fusion: combines two ranked lists using only RANK
    POSITION, never raw score magnitude. RRF(d) = sum over each ranking of
    1/(k + rank(d)), where rank is 1-indexed and a document absent from a
    ranking contributes nothing from that side.

    Why this is worth having as an alternative to the weighted min-max fusion
    already in hybrid_search: that approach is vulnerable to exactly the
    failure mode found and fixed earlier in this project -- min-max
    normalization is relative to the candidate pool, so a weak signal can
    look artificially strong after normalization (this is why confidence
    gating had to be moved to raw_dense_similarity instead of the fused
    score). RRF sidesteps that class of problem entirely since it never
    touches score magnitude at all, only ordinal position -- a common
    standard choice in production hybrid retrieval for that reason.

    k=60 is the constant from the original RRF paper (Cormack et al.) and is
    a conventional default, not tuned specifically for this corpus.
    """
    ranked_dense = sorted(dense_scores, key=dense_scores.get, reverse=True)
    ranked_sparse = sorted(sparse_scores, key=sparse_scores.get, reverse=True)

    dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(ranked_dense, start=1)}
    sparse_ranks = {chunk_id: rank for rank, chunk_id in enumerate(ranked_sparse, start=1)}

    all_ids = set(dense_scores) | set(sparse_scores)
    rrf_scores = {}
    for chunk_id in all_ids:
        score = 0.0
        if chunk_id in dense_ranks:
            score += 1.0 / (k + dense_ranks[chunk_id])
        if chunk_id in sparse_ranks:
            score += 1.0 / (k + sparse_ranks[chunk_id])
        rrf_scores[chunk_id] = score

    return rrf_scores


def hybrid_search(
    query: str,
    corpus: list[dict],
    corpus_by_id: dict[str, dict],
    bm25,
    collection,
    embedder,
    settings: dict,
    exclude_ads: bool = False,
    cross_encoder=None,
) -> list[dict]:
    r = settings["retrieval"]
    top_k = r["top_k"]
    dense_weight = r["dense_weight"]
    bm25_weight = r["bm25_weight"]
    fusion_method = r.get("fusion_method", "weighted")

    # Pull a wider candidate pool from each side than top_k, so the merge has
    # enough overlap/signal to work with before truncating to top_k. This
    # same pool is what the reranker (if enabled) reorders before the final
    # truncation below -- reusing it rather than doing a second retrieval pass.
    pool_size = max(top_k * 4, 20)

    dense_scores, dense_raw = dense_search(collection, embedder, query, pool_size)
    sparse_scores = bm25_search(bm25, corpus, query, pool_size)

    rrf_scores = None
    if fusion_method == "rrf":
        rrf_scores = rrf_fusion(dense_scores, sparse_scores, k=r.get("rrf_k", 60))

    all_ids = set(dense_scores) | set(sparse_scores)
    combined = []
    for chunk_id in all_ids:
        record = corpus_by_id.get(chunk_id)
        if not record:
            continue
        if exclude_ads and record["metadata"].get("is_likely_ad"):
            continue

        d_score = dense_scores.get(chunk_id, 0.0)
        s_score = sparse_scores.get(chunk_id, 0.0)

        if fusion_method == "rrf":
            final_score = rrf_scores[chunk_id]
        else:
            final_score = dense_weight * d_score + bm25_weight * s_score

        combined.append(
            {
                "chunk_id": chunk_id,
                "score": final_score,
                "dense_score": d_score,
                "bm25_score": s_score,
                "raw_dense_similarity": dense_raw.get(chunk_id, 0.0),
                "text": record["text"],
                "episode_id": record["episode_id"],
                "start": record["start"],
                "end": record["end"],
                "metadata": record["metadata"],
            }
        )

    use_reranker = r.get("use_reranker", False)
    if use_reranker and cross_encoder is not None and combined:
        combined = rerank_pool(query, combined, cross_encoder)
    else:
        combined.sort(key=lambda r: r["score"], reverse=True)

    return combined[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Query the hybrid dense + BM25 retriever")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-chunks"),
    )
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--top-k", type=int, default=None, help="Override settings.yaml top_k")
    parser.add_argument(
        "--exclude-ads", action="store_true", help="Filter out chunks flagged as likely ads"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable the cross-encoder reranker for this query, for A/B comparison",
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        choices=["weighted", "rrf"],
        default=None,
        help="Override settings.yaml retrieval.fusion_method, for A/B comparison",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    if args.top_k:
        settings["retrieval"]["top_k"] = args.top_k
    if args.no_rerank:
        settings["retrieval"]["use_reranker"] = False
    if args.fusion_method:
        settings["retrieval"]["fusion_method"] = args.fusion_method

    log.info("Loading corpus from %s", args.chunks)
    corpus = load_corpus(args.chunks)
    if not corpus:
        log.error("No chunks found in %s -- run ingest/chunk.py first.", args.chunks)
        sys.exit(1)
    corpus_by_id = {r["chunk_id"]: r for r in corpus}
    log.info("Corpus loaded: %d chunks", len(corpus))

    log.info("Building BM25 index...")
    bm25 = build_bm25(corpus)

    log.info("Loading embedder and Chroma collection...")
    embedder = get_embedder(settings)
    collection = get_collection(settings)

    cross_encoder = None
    if settings["retrieval"].get("use_reranker", False):
        log.info("Loading cross-encoder reranker...")
        cross_encoder = get_cross_encoder(settings)

    results = hybrid_search(
        args.query,
        corpus,
        corpus_by_id,
        bm25,
        collection,
        embedder,
        settings,
        args.exclude_ads,
        cross_encoder=cross_encoder,
    )

    print(f"\nTop {len(results)} results for: {args.query!r}\n")
    for i, r in enumerate(results, 1):
        rerank_info = f" rerank={r['rerank_score']:.3f}" if "rerank_score" in r else ""
        print(
            f"--- {i}. score={r['score']:.3f} (dense={r['dense_score']:.3f}, "
            f"bm25={r['bm25_score']:.3f}){rerank_info} guest={r['metadata'].get('guest')} "
            f"is_ad={r['metadata'].get('is_likely_ad')} t={r['start']:.0f}-{r['end']:.0f}s ---"
        )
        print(r["text"][:200])
        print()


if __name__ == "__main__":
    main()
