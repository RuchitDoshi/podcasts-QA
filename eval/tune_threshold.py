"""
Diagnostic for picking retrieval.min_score_threshold empirically instead of
guessing. Runs the scored queries from eval/queries.yaml through DENSE
search only (not the fused hybrid score -- see agent/tools.py's docstring
for why raw dense similarity is the only number in this pipeline with
genuine absolute meaning), and prints the raw_dense_similarity of:

  - the correct-episode result, for every query with a known answer
  - a deliberately irrelevant control query, as a negative reference point

The gap between those two groups of numbers is what actually justifies a
threshold value -- bge-small's baseline similarity for unrelated text varies
by domain, so a threshold picked without looking at real data is a guess,
not a decision.

Usage:
    python eval/tune_threshold.py
    python eval/tune_threshold.py --control-query "banana bread recipe"
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieve.hybrid import get_collection, get_embedder, load_corpus  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tune_threshold")

DEFAULT_CONTROL_QUERIES = [
    "banana bread recipe ingredients",
    "how to change a car tire",
    "best vacation spots in the Caribbean",
]


def load_queries(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f).get("queries", [])


def probe_query(
    query: str, collection, embedder, expected_episode_id: str | None, n_results: int = 10
):
    """Runs dense-only search and returns (matched, raw_similarity, episode_id)
    for either the expected episode's best-ranked appearance, or the top
    result if there's no expected episode (control queries)."""
    query_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=n_results)

    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    similarities = [1 - d for d in distances]

    if expected_episode_id:
        for i, meta in enumerate(metadatas):
            if meta.get("episode_id") == expected_episode_id:
                return True, similarities[i], expected_episode_id, i + 1
        return False, None, None, None
    else:
        # Control query: no "correct" answer exists -- report the TOP result,
        # since that's the score a min_score_threshold would actually have to
        # reject to correctly produce a no_relevant_results signal.
        return True, similarities[0], metadatas[0].get("episode_id"), 1


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose a real min_score_threshold value from actual corpus data"
    )
    parser.add_argument("--queries", type=Path, default=Path("eval/queries.yaml"))
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-chunks"),
    )
    parser.add_argument(
        "--control-query",
        type=str,
        action="append",
        default=None,
        help="An irrelevant query with no real answer in the corpus. Repeatable. "
        "Defaults to a few generic off-topic queries if not given.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        settings = yaml.safe_load(f)

    log.info("Loading corpus (for episode_id lookups only, not re-embedding)...")
    corpus = load_corpus(args.chunks)
    if not corpus:
        log.error("No chunks found in %s -- run ingest/chunk.py first.", args.chunks)
        sys.exit(1)

    embedder = get_embedder(settings)
    collection = get_collection(settings)

    query_specs = [q for q in load_queries(args.queries) if q.get("expected_episode_id")]
    control_queries = args.control_query or DEFAULT_CONTROL_QUERIES

    print("\n" + "=" * 70)
    print("KNOWN-ANSWER QUERIES -- raw dense similarity of the correct episode")
    print("=" * 70)
    positive_scores = []
    for spec in query_specs:
        matched, sim, ep_id, rank = probe_query(
            spec["query"], collection, embedder, spec["expected_episode_id"]
        )
        if matched:
            positive_scores.append(sim)
            print(f"  {sim:.3f}  (rank {rank})  {spec['query']}")
        else:
            print(f"  MISS -- correct episode not in top-10  {spec['query']}")

    print("\n" + "=" * 70)
    print("CONTROL QUERIES (no real answer in corpus) -- raw similarity of the TOP result")
    print("=" * 70)
    negative_scores = []
    for cq in control_queries:
        _, sim, ep_id, _ = probe_query(cq, collection, embedder, None)
        negative_scores.append(sim)
        print(f"  {sim:.3f}  top_episode={ep_id}  {cq}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if positive_scores:
        print(
            f"Known-answer scores: min={min(positive_scores):.3f} max={max(positive_scores):.3f} "
            f"avg={sum(positive_scores) / len(positive_scores):.3f}"
        )
    if negative_scores:
        print(
            f"Control (irrelevant) scores: min={min(negative_scores):.3f} max={max(negative_scores):.3f} "
            f"avg={sum(negative_scores) / len(negative_scores):.3f}"
        )

    if positive_scores and negative_scores:
        gap_low = min(positive_scores)
        gap_high = max(negative_scores)
        if gap_low > gap_high:
            suggested = round((gap_low + gap_high) / 2, 2)
            print(
                f"\nClean separation: lowest known-answer score ({gap_low:.3f}) is above "
                f"highest control score ({gap_high:.3f})."
            )
            print(f"Suggested min_score_threshold: {suggested} (midpoint of the gap)")
        else:
            print(
                f"\nNo clean separation: lowest known-answer score ({gap_low:.3f}) is BELOW "
                f"highest control score ({gap_high:.3f})."
            )
            print(
                "A single global threshold can't perfectly separate these with the current data -- "
                "consider more control queries, or accept some overlap and pick a threshold that "
                "trades off which side you'd rather err on."
            )


if __name__ == "__main__":
    main()
