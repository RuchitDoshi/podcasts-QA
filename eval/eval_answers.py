"""
Evaluates answer QUALITY, not just retrieval -- runs the fixed query set
through the real agent (agent/orchestrator.py) end to end, then judges each
resulting answer against the chunks it actually cited, using Gemini as the
judge (a different model family than the Groq-based answerer, to avoid
self-preference bias -- the same principle behind separating answer and
judge models throughout this project).

This is deliberately layered on top of, not a replacement for, the
mechanical citation verification already built into the orchestrator:
- Citation verification (orchestrator.py) checks a cheap, code-level fact:
  was this chunk_id/episode_id/timestamp actually shown to the model this
  session? It catches fabricated citations but NOT subtler misrepresentation
  of a correctly-cited chunk's content -- a model can cite a real chunk and
  still describe it inaccurately, and nothing before this script would catch
  that.
- Faithfulness (this script, LLM-judged) checks the harder, semantic
  question: does the answer's actual content match what the cited chunks
  say? This is the gap named repeatedly earlier in this project
  ("citation verification proves a chunk was retrieved, not that the
  answer about it is correct") and left as future work until now.
- Relevance (this script, LLM-judged) checks a question mechanical
  verification can't touch at all: does the answer actually address what
  was asked, or does it wander/hedge/answer a different question.

Usage:
    python eval/eval_answers.py
    python eval/eval_answers.py --queries eval/queries.yaml --output eval/results/answers.json
"""

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.orchestrator import AgentContext, get_groq_client, run_agent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_answers")

JUDGE_SYSTEM_PROMPT = """You evaluate whether an AI assistant's answer about a podcast \
is faithful to the source excerpts it cited, and whether it actually addresses the \
question that was asked.

Score two dimensions, 1-5 each:

faithfulness: does the answer's content accurately reflect what the cited excerpts \
actually say, with no claims that are unsupported, exaggerated, or contradicted by \
them? 5 = every claim is directly supported. 3 = mostly supported but some overstatement \
or imprecision. 1 = contains claims not found in or contradicted by the excerpts.

relevance: does the answer actually address the question asked? 5 = directly and \
completely answers it. 3 = partially answers or hedges more than necessary. \
1 = off-topic or non-responsive.

Respond with only a JSON object: {"faithfulness": <int 1-5>, "relevance": <int 1-5>, \
"notes": "<one brief sentence explaining the scores>"}"""


def _is_rate_limit_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status in (429, "429"):
        return True
    msg = str(error).lower()
    return "rate limit" in msg or "resourceexhausted" in msg or "quota" in msg


def get_gemini_client(model: str):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        import google.generativeai as genai
    except ImportError:
        log.error("google-generativeai not installed. Run: pip install google-generativeai")
        sys.exit(1)

    import os

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("GOOGLE_API_KEY not set -- required for the Gemini judge.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model, system_instruction=JUDGE_SYSTEM_PROMPT)


def judge_answer(
    query: str, answer: str, cited_chunks_text: list[str], judge_client, max_retries: int = 4
) -> dict | None:
    """Returns {"faithfulness": int, "relevance": int, "notes": str}, or None
    if judging failed/was unparseable -- callers should treat None as "not
    scored" and exclude from aggregates rather than silently zero-filling,
    since a missing score is a different thing from a bad score."""
    if cited_chunks_text:
        excerpts = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(cited_chunks_text))
    else:
        excerpts = "(none -- the answer cited no source excerpts)"

    prompt = f"Question: {query}\n\nAnswer given: {answer}\n\nCited source excerpts:\n{excerpts}"

    for attempt in range(max_retries):
        try:
            response = judge_client.generate_content(prompt)
            raw = response.text.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)

            if (
                not isinstance(result, dict)
                or "faithfulness" not in result
                or "relevance" not in result
            ):
                log.warning("Judge returned unexpected shape: %r", raw)
                return None

            return {
                "faithfulness": int(result["faithfulness"]),
                "relevance": int(result["relevance"]),
                "notes": str(result.get("notes", "")),
            }

        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries - 1:
                wait = 8.0 * (attempt + 1)
                log.warning(
                    "Judge rate limited, attempt %d/%d -- waiting %.0fs",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue
            log.warning("Judging failed (%s) -- excluding this query from aggregates", e)
            return None

    return None


def load_queries(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f).get("queries", [])


def run_eval(
    query_specs: list[dict],
    ctx: AgentContext,
    groq_client,
    judge_client,
    agent_model: str,
    sleep_between_judge_calls: float = 4.5,
    sleep_between_queries: float = 3.0,
) -> list[dict]:
    """Pacing note: each query runs the full agent loop, which can itself be
    2-4+ Groq calls (list_episodes, search_transcripts, submit_answer, plus
    any retries). call_llm_with_retry now handles rate limits reactively,
    but running many queries back-to-back with zero pacing is exactly the
    pattern that triggers them in the first place -- proactive spacing
    between queries reduces how often that reactive path even needs to
    fire. sleep_between_judge_calls covers the separate Gemini rate limit."""
    results = []
    for i, spec in enumerate(query_specs):
        query = spec["query"]
        log.info("Running agent for: %s", query)
        agent_result = run_agent(query, ctx, groq_client, agent_model)

        row = {
            "query": query,
            "type": spec.get("type", "clean"),
            "has_answer": agent_result["has_answer"],
            "answer": agent_result["answer"],
            "mechanically_grounded": agent_result["grounded"],
            "all_citations_verified": agent_result["all_citations_verified"],
            "num_citations": len(agent_result["citations"]),
            "faithfulness": None,
            "relevance": None,
            "judge_notes": None,
        }

        if agent_result["has_answer"] and agent_result["citations"]:
            cited_texts = [
                ctx.corpus_by_id[c["chunk_id"]]["text"]
                for c in agent_result["citations"]
                if c.get("verified") and c["chunk_id"] in ctx.corpus_by_id
            ]
            judgment = judge_answer(query, agent_result["answer"], cited_texts, judge_client)
            if judgment:
                row["faithfulness"] = judgment["faithfulness"]
                row["relevance"] = judgment["relevance"]
                row["judge_notes"] = judgment["notes"]
            time.sleep(sleep_between_judge_calls)
        else:
            log.info("Skipping judging for '%s' -- no answer or no citations to check", query)

        results.append(row)

        if i < len(query_specs) - 1:
            time.sleep(sleep_between_queries)

    return results


def print_report(results: list[dict]):
    judged = [r for r in results if r["faithfulness"] is not None]
    declined = [r for r in results if not r["has_answer"]]

    print("\n" + "=" * 70)
    print("PER-QUERY RESULTS")
    print("=" * 70)
    for r in results:
        if r["faithfulness"] is not None:
            print(f"[{r['type']}] {r['query']}")
            print(
                f"    faithfulness={r['faithfulness']}/5  relevance={r['relevance']}/5  "
                f"citations={r['num_citations']}  verified={r['all_citations_verified']}"
            )
            print(f"    judge notes: {r['judge_notes']}")
        elif not r["has_answer"]:
            print(f"[{r['type']}] {r['query']}")
            print("    -> declined to answer (has_answer=False)")
        else:
            print(f"[{r['type']}] {r['query']}")
            print("    -> answered but not judged (no verified citations to check)")
        print()

    if judged:
        avg_faith = sum(r["faithfulness"] for r in judged) / len(judged)
        avg_rel = sum(r["relevance"] for r in judged) / len(judged)
        grounded_rate = sum(r["all_citations_verified"] for r in judged) / len(judged)
        print(f"--- Aggregate (judged queries only, n={len(judged)}) ---")
        print(f"avg faithfulness: {avg_faith:.2f}/5")
        print(f"avg relevance:    {avg_rel:.2f}/5")
        print(f"citation verification rate: {grounded_rate:.1%}")

    print(f"\nDeclined to answer: {len(declined)}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent answer quality via LLM-as-judge")
    parser.add_argument("--queries", type=Path, default=Path("eval/queries.yaml"))
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-chunks"),
    )
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--episodes-config", type=Path, default=Path("config/episodes.yaml"))
    parser.add_argument("--agent-model", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-flash-latest",
        help="Alias that Google keeps pointed at the current recommended Flash model -- "
        "avoids breaking again when a specific dated model (e.g. gemini-2.5-flash) gets "
        "gated from new API users, as happened during this project.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        settings = yaml.safe_load(f)

    query_specs = load_queries(args.queries)
    if not query_specs:
        log.error("No queries found in %s", args.queries)
        sys.exit(1)

    ctx = AgentContext(settings, args.chunks, args.episodes_config)
    groq_client = get_groq_client()
    judge_client = get_gemini_client(args.judge_model)

    results = run_eval(query_specs, ctx, groq_client, judge_client, args.agent_model)
    print_report(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"timestamp": datetime.now(UTC).isoformat(), "results": results}, f, indent=2)
        log.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
