"""
Builds a side-by-side comparison table across multiple eval_answers.py runs
(one per provider/model), reading their saved --output JSON files. Answers
the question this project's multi-provider work has been building toward:
given real faithfulness/relevancy/correctness/cost/latency numbers, which
model is actually the better choice, not just which one happened to work.

Usage:
    python eval/eval_answers.py --provider groq --output eval/results/answers_groq.json
    python eval/eval_answers.py --provider litellm --model "openai.gpt-oss-120b-1:0" --output eval/results/answers_gptoss.json
    python eval/eval_answers.py --provider litellm --model "minimax.minimax-m2.5" --output eval/results/answers_minimax.json
    python eval/compare_providers.py eval/results/answers_groq.json eval/results/answers_gptoss.json eval/results/answers_minimax.json
"""

import argparse
import json
from pathlib import Path


def load_run(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data


def summarize_run(data: dict) -> dict:
    results = data["results"]

    def avg(key_path):
        vals = []
        for r in results:
            obj = r
            for k in key_path[:-1]:
                obj = obj.get(k) if obj else None
            if obj and obj.get(key_path[-1]) is not None:
                vals.append(obj[key_path[-1]])
        return sum(vals) / len(vals) if vals else None

    n_declined = sum(1 for r in results if not r["has_answer"])
    with_cost = [r for r in results if r.get("dollar_cost_estimate") is not None]
    avg_cost = (
        sum(r["dollar_cost_estimate"] for r in with_cost) / len(with_cost) if with_cost else None
    )

    return {
        "provider": data.get("provider", "?"),
        "model": data.get("model", "?"),
        "n_queries": len(results),
        "n_declined": n_declined,
        "avg_faithfulness": avg(["faithfulness", "score"]),
        "avg_relevancy": avg(["relevancy", "score"]),
        "avg_correctness": avg(["correctness", "score"]),
        "avg_tokens": sum(r["tokens_total"] for r in results) / len(results) if results else None,
        "avg_latency_s": sum(r["latency_s"] for r in results) / len(results) if results else None,
        "avg_cost": avg_cost,
    }


def fmt(v, spec=".3f"):
    return format(v, spec) if v is not None else "N/A"


def print_comparison(summaries: list[dict]):
    print("\n" + "=" * 100)
    print("PROVIDER / MODEL COMPARISON")
    print("=" * 100)

    col_width = 24
    header = f"{'Metric':<28}" + "".join(
        f"{s['model'][: col_width - 1]:<{col_width}}" for s in summaries
    )
    print(header)
    print("-" * len(header))

    rows = [
        ("Provider", lambda s: s["provider"]),
        ("Queries run", lambda s: str(s["n_queries"])),
        ("Declined to answer", lambda s: f"{s['n_declined']}/{s['n_queries']}"),
        ("Faithfulness (avg)", lambda s: fmt(s["avg_faithfulness"])),
        ("Relevancy (avg)", lambda s: fmt(s["avg_relevancy"])),
        ("Correctness (avg)", lambda s: fmt(s["avg_correctness"])),
        ("Avg tokens/task", lambda s: fmt(s["avg_tokens"], ".0f")),
        ("Avg latency (s)", lambda s: fmt(s["avg_latency_s"], ".2f")),
        ("Avg cost/task ($)", lambda s: fmt(s["avg_cost"], ".5f")),
    ]
    for label, fn in rows:
        print(f"{label:<28}" + "".join(f"{fn(s):<{col_width}}" for s in summaries))

    print()
    scored = [s for s in summaries if s["avg_faithfulness"] is not None]
    if scored:
        best_faithfulness = max(scored, key=lambda s: s["avg_faithfulness"])
        print(
            f"Highest faithfulness: {best_faithfulness['model']} ({best_faithfulness['avg_faithfulness']:.3f})"
        )
    cost_scored = [s for s in summaries if s["avg_cost"] is not None]
    if cost_scored:
        cheapest = min(cost_scored, key=lambda s: s["avg_cost"])
        print(f"Cheapest per task:    {cheapest['model']} (${cheapest['avg_cost']:.5f})")
    fastest_candidates = [s for s in summaries if s["avg_latency_s"] is not None]
    if fastest_candidates:
        fastest = min(fastest_candidates, key=lambda s: s["avg_latency_s"])
        print(f"Fastest:              {fastest['model']} ({fastest['avg_latency_s']:.2f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple eval_answers.py runs side by side"
    )
    parser.add_argument(
        "runs", type=Path, nargs="+", help="Paths to saved eval_answers.py --output JSON files"
    )
    args = parser.parse_args()

    summaries = [summarize_run(load_run(p)) for p in args.runs]
    print_comparison(summaries)


if __name__ == "__main__":
    main()
