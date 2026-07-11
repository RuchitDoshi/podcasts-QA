"""
Run a fixed query set (eval/queries.yaml) through hybrid retrieval and report
aggregate metrics -- this replaces one-off manual query inspection with a
repeatable, numeric baseline you can compare against after any pipeline
change (chunking strategy, weights, ad filtering, etc.).

Metrics computed, per query with a known expected_episode_id:
- hit@1: does the top result come from the expected episode?
- precision@k: what fraction of the top-k results come from the expected
  episode? (a coarse proxy for relevance -- see queries.yaml header)
- retrieval null rate: did NO result in the top-k come from the expected
  episode at all? (the metric your architecture doc's eval plan calls out
  specifically)

For queries with no expected_episode_id (ambiguous/cross-episode/no-answer
by design), no pass/fail is scored -- results are shown for manual read,
since "no good answer exists" isn't something precision@k can capture.

Ad leakage is reported separately across ALL queries regardless of type:
% of top-k results flagged is_likely_ad. Run once with --exclude-ads off
(baseline) and once with it on, to get a real before/after number instead
of eyeballing single queries.

Usage:
    python eval/eval_retrieval.py
    python eval/eval_retrieval.py --exclude-ads
    python eval/eval_retrieval.py --top-k 5
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieve.hybrid import (  # noqa: E402
    build_bm25,
    get_collection,
    get_cross_encoder,
    get_embedder,
    hybrid_search,
    load_corpus,
    load_settings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_retrieval")


def load_queries(path: Path) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("queries", [])


def evaluate_query(query_spec: dict, results: list[dict]) -> dict:
    expected = query_spec.get("expected_episode_id")
    ad_count = sum(1 for r in results if r["metadata"].get("is_likely_ad"))
    ad_rate = ad_count / len(results) if results else 0.0

    metrics = {
        "query": query_spec["query"],
        "type": query_spec.get("type", "clean"),
        "expected_episode_id": expected,
        "num_results": len(results),
        "ad_rate": round(ad_rate, 3),
        "top_result_episode": results[0]["episode_id"] if results else None,
    }

    if expected:
        hit_at_1 = bool(results) and results[0]["episode_id"] == expected
        matching = sum(1 for r in results if r["episode_id"] == expected)
        precision_at_k = matching / len(results) if results else 0.0
        null_rate = matching == 0

        # Reciprocal rank: 1/(rank of first correct-episode result), 0 if none
        # in top-k. Generalizes hit@1 -- hit@1 is really just "was RR == 1.0",
        # so this is free to compute from data we already have (the ranked
        # results list), no new retrieval calls or labels needed.
        reciprocal_rank = 0.0
        for rank, r in enumerate(results, start=1):
            if r["episode_id"] == expected:
                reciprocal_rank = 1.0 / rank
                break

        metrics.update(
            {
                "hit_at_1": hit_at_1,
                "precision_at_k": round(precision_at_k, 3),
                "retrieval_null": null_rate,
                "reciprocal_rank": round(reciprocal_rank, 3),
            }
        )

    return metrics


def print_report(all_metrics: list[dict]):
    scored = [m for m in all_metrics if m.get("expected_episode_id")]
    unscored = [m for m in all_metrics if not m.get("expected_episode_id")]

    print("\n" + "=" * 70)
    print("SCORED QUERIES (have a known expected episode)")
    print("=" * 70)
    for m in scored:
        status = "PASS" if m["hit_at_1"] else "FAIL"
        print(f"[{status}] {m['query']}")
        print(
            f"    hit@1={m['hit_at_1']}  precision@k={m['precision_at_k']:.2f}  "
            f"RR={m['reciprocal_rank']:.2f}  null={m['retrieval_null']}  "
            f"ad_rate={m['ad_rate']:.2f}  type={m['type']}"
        )

    if scored:
        hit_rate = sum(m["hit_at_1"] for m in scored) / len(scored)
        avg_precision = sum(m["precision_at_k"] for m in scored) / len(scored)
        null_rate = sum(m["retrieval_null"] for m in scored) / len(scored)
        avg_ad_rate = sum(m["ad_rate"] for m in scored) / len(scored)
        mrr = sum(m["reciprocal_rank"] for m in scored) / len(scored)
        print("\n--- Aggregate (scored queries) ---")
        print(f"hit@1 rate:        {hit_rate:.1%}")
        print(f"avg precision@k:   {avg_precision:.1%}")
        print(f"MRR:               {mrr:.3f}")
        print(f"retrieval null rate: {null_rate:.1%}")
        print(f"avg ad rate:       {avg_ad_rate:.1%}")

    print("\n" + "=" * 70)
    print("UNSCORED QUERIES (ambiguous / cross-episode / no correct answer)")
    print("=" * 70)
    for m in unscored:
        print(f"{m['query']}")
        print(f"    top_result_episode={m['top_result_episode']}  ad_rate={m['ad_rate']:.2f}")

    if all_metrics:
        overall_ad_rate = sum(m["ad_rate"] for m in all_metrics) / len(all_metrics)
        print(f"\nOverall ad rate across ALL {len(all_metrics)} queries: {overall_ad_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid retrieval against a fixed query set"
    )
    parser.add_argument("--queries", type=Path, default=Path("eval/queries.yaml"))
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-chunks"),
    )
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--exclude-ads", action="store_true")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable the cross-encoder reranker, for A/B comparison against retrieval.use_reranker",
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        choices=["weighted", "rrf"],
        default=None,
        help="Override settings.yaml retrieval.fusion_method, for A/B comparison",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to save results as JSON"
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    if args.top_k:
        settings["retrieval"]["top_k"] = args.top_k
    if args.no_rerank:
        settings["retrieval"]["use_reranker"] = False
    if args.fusion_method:
        settings["retrieval"]["fusion_method"] = args.fusion_method

    query_specs = load_queries(args.queries)
    if not query_specs:
        log.error("No queries found in %s", args.queries)
        sys.exit(1)

    log.info("Loading corpus from %s", args.chunks)
    corpus = load_corpus(args.chunks)
    corpus_by_id = {r["chunk_id"]: r for r in corpus}
    log.info("Corpus loaded: %d chunks", len(corpus))

    bm25 = build_bm25(corpus)
    embedder = get_embedder(settings)
    collection = get_collection(settings)

    cross_encoder = None
    if settings["retrieval"].get("use_reranker", False):
        log.info("Loading cross-encoder reranker...")
        cross_encoder = get_cross_encoder(settings)

    all_metrics = []
    for spec in query_specs:
        results = hybrid_search(
            spec["query"],
            corpus,
            corpus_by_id,
            bm25,
            collection,
            embedder,
            settings,
            exclude_ads=args.exclude_ads,
            cross_encoder=cross_encoder,
        )
        all_metrics.append(evaluate_query(spec, results))

    print_report(all_metrics)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "exclude_ads": args.exclude_ads,
            "top_k": settings["retrieval"]["top_k"],
            "results": all_metrics,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
