"""
Evaluates answer QUALITY, not just retrieval or trajectory -- runs the
scored query set through the real agent end to end, then scores each
answer on three RAGAS-style dimensions, plus cost and latency.

- faithfulness: does the answer's content match what the cited chunks say?
  Claim-level, not holistic -- the judge decomposes the answer into atomic
  factual claims and checks each independently against the cited excerpts;
  score = supported_claims / total_claims. This replaced an earlier
  holistic 1-5 judge score specifically because a single number couldn't
  distinguish "one bad hallucinated number in an otherwise correct answer"
  from "systematically wrong" -- two real cases seen in this project
  (Jensen Huang's "4 trillion" vs. the source's "3 trillion", and Dan
  Houser's answer that both fabricated and omitted real content) scored
  identically under the old holistic approach.

- answer_relevancy: does the answer actually address the question asked?
  RAGAS's real approach, not a judge guess: generate several plausible
  questions the answer WOULD be a good answer to, embed them with this
  project's own retrieval embedder (bge-small, already loaded, no new
  model), and measure cosine similarity against the actual question. A low
  score means the answer wandered, was incomplete, or padded with
  tangential content -- this is genuinely cheaper than an LLM judge call
  for the scoring step itself (only the question-generation step needs the
  judge).

- answer_correctness: does the answer match a reference answer? Needs a
  reference, which this project doesn't have as human-labeled ground
  truth -- eval/generate_references.py produces LLM-generated
  ("silver") references from the same validated-correct chunks used
  elsewhere in this project. Scored as claim-overlap F1 (precision/recall
  over reference claims found in the candidate vs. extra unsupported
  claims in the candidate) -- the factual core of RAGAS's actual
  methodology, computed deterministically from the judge's claim
  extraction rather than trusting the judge's own arithmetic.

This is deliberately layered on top of, not a replacement for, the
mechanical citation verification already built into the orchestrator:
citation verification proves a chunk was actually shown to the model;
faithfulness checks whether the answer accurately describes it, which is a
different and harder question mechanical checks can't touch at all.

Usage:
    python eval/eval_answers.py
    python eval/eval_answers.py --provider litellm --model "openai.gpt-oss-120b-1:0"
    python eval/eval_answers.py --references eval/reference_answers.yaml --output eval/results/answers.json
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
from agent.orchestrator import (  # noqa: E402
    PROVIDER_DEFAULT_MODELS,
    AgentContext,
    get_client_for_provider,
    run_agent,
)
from eval.cost_utils import estimate_dollar_cost  # noqa: E402
from retrieve.hybrid import get_embedder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_answers")

FAITHFULNESS_SYSTEM_PROMPT = """You evaluate whether an AI-generated answer is \
faithful to source excerpts, at the level of individual factual claims -- not a \
single holistic judgment.

1. Break the ANSWER into a list of atomic factual claims. Each claim should be a \
single, standalone factual statement (not opinion, not filler like "he explains that").
2. For each claim, decide whether it is DIRECTLY supported by the source excerpts: \
true if the excerpts state or clearly imply it, false if it's unsupported, \
exaggerated, or contradicted.

Respond with only JSON: {"claims": [{"claim": "...", "supported": true|false}, ...]}"""

RELEVANCY_SYSTEM_PROMPT = """Given an answer, generate 3 different questions that \
this answer would be a good, complete response to. Base the questions ONLY on what \
the answer actually discusses -- do not invent topics the answer doesn't cover. \
Phrase them naturally, as a curious person would ask them.

Respond with only JSON: {"questions": ["...", "...", "..."]}"""

CORRECTNESS_SYSTEM_PROMPT = """You compare a candidate answer against a reference \
answer for factual overlap, at the level of individual claims.

1. List the atomic factual claims in the REFERENCE answer.
2. For each reference claim, decide whether it is ALSO present in the CANDIDATE \
answer (true/false).
3. Separately, list any claims in the CANDIDATE that are NOT supported by the \
reference (extra or contradicting claims not grounded in the reference).

Respond with only JSON: {"reference_claims": [{"claim": "...", \
"present_in_candidate": true|false}, ...], "candidate_extra_claims": ["...", ...]}"""


def _is_rate_limit_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status in (429, "429"):
        return True
    msg = str(error).lower()
    return "rate limit" in msg or "resourceexhausted" in msg or "quota" in msg


def _is_daily_limit_error(error: Exception) -> bool:
    """Same lesson learned from Groq's TPD limit (agent/orchestrator.py) --
    a DAILY quota (clears in hours) needs a completely different response
    than a per-minute rate limit (clears in seconds): retrying with a short
    backoff just wastes time proving what the error already said. Found in
    practice with the ORIGINAL Gemini judge setup: gemini-flash-latest's
    free tier turned out to cap at 20 requests/DAY, and the claim-level
    judge rebuild uses up to 3 calls/query -- nowhere near enough headroom
    for a real eval run, which is the actual reason this project switched
    the default judge to a LiteLLM-hosted model instead of just retrying
    harder against Gemini's tiny daily cap."""
    msg = str(error).lower()
    return "per day" in msg or "daily" in msg or "generaterequestsperdayper" in msg.replace("_", "")


class _GenerateContentAdapter:
    """Wraps an OpenAI-style chat client (client.chat.completions.create(...),
    as exposed by agent.orchestrator.get_client_for_provider for both the
    groq and litellm providers) to expose the same .generate_content(prompt)
    -> response.text interface the judge_* functions below already call --
    so switching the judge's underlying provider doesn't require touching
    any of the actual judging logic, same "isolate the change in an
    adapter" approach already used for the LangChain/LiteLLM agent client."""

    def __init__(self, chat_client, model: str):
        self._client = chat_client
        self._model = model

    def generate_content(self, prompt: str):
        from types import SimpleNamespace

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""
        return SimpleNamespace(text=text)


def get_judge_client(provider: str, model: str):
    """Builds a judge client exposing .generate_content(prompt).text
    regardless of provider -- "gemini" uses the native google-generativeai
    client directly; anything else goes through agent.orchestrator's
    existing provider dispatcher (groq/litellm) wrapped in
    _GenerateContentAdapter."""
    if provider == "gemini":
        return get_gemini_client(model)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent.orchestrator import get_client_for_provider

    chat_client = get_client_for_provider(provider)
    return _GenerateContentAdapter(chat_client, model)


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
    return genai.GenerativeModel(model)


def _call_judge_json(
    judge_client, system_prompt: str, user_prompt: str, max_retries: int = 4
) -> dict | list | None:
    """Shared retry/parse logic for all three judge calls -- returns the
    parsed JSON (dict or list depending on the prompt's schema), or None if
    every attempt failed to produce parseable output."""
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    for attempt in range(max_retries):
        try:
            response = judge_client.generate_content(full_prompt)
            raw = response.text.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning("Judge returned unparseable JSON: %s", e)
            return None
        except Exception as e:
            if _is_daily_limit_error(e):
                log.error(
                    "Judge hit a DAILY quota limit -- not retrying (would need to wait far "
                    "longer than this function's retry budget, likely hours not seconds): %s",
                    e,
                )
                e.is_daily_limit = True
                raise

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
            log.warning("Judge call failed (%s)", e)
            return None
    return None


def judge_faithfulness(
    query: str, answer: str, cited_chunks_text: list[str], judge_client
) -> dict | None:
    """Returns {"score": float 0-1, "claims": [{"claim":str,"supported":bool}],
    "num_claims": int, "num_supported": int}, or None if judging failed.
    score is computed deterministically from the claims list (supported/total),
    not trusted from the judge's own arithmetic."""
    excerpts = (
        "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(cited_chunks_text))
        if cited_chunks_text
        else "(none -- the answer cited no source excerpts)"
    )
    user_prompt = f"Question: {query}\n\nAnswer: {answer}\n\nSource excerpts:\n{excerpts}"

    result = _call_judge_json(judge_client, FAITHFULNESS_SYSTEM_PROMPT, user_prompt)
    if (
        not isinstance(result, dict)
        or "claims" not in result
        or not isinstance(result["claims"], list)
    ):
        log.warning("Faithfulness judge returned unexpected shape: %r", result)
        return None

    claims = result["claims"]
    if not claims:
        # No claims extracted (e.g. a very short/empty answer) -- treat as
        # vacuously faithful (nothing unsupported was said) rather than
        # dividing by zero or silently excluding.
        return {"score": 1.0, "claims": [], "num_claims": 0, "num_supported": 0}

    num_supported = sum(1 for c in claims if c.get("supported") is True)
    return {
        "score": num_supported / len(claims),
        "claims": claims,
        "num_claims": len(claims),
        "num_supported": num_supported,
    }


def judge_relevancy(
    query: str, answer: str, judge_client, embedder, n_questions: int = 3
) -> dict | None:
    """Returns {"score": float 0-1, "generated_questions": [str, ...]}, or
    None if judging failed. Score is the average cosine similarity between
    the real question and each LLM-generated hypothetical question the
    answer would address -- computed locally via this project's own
    embedder, not by the judge."""
    if not answer or not answer.strip():
        return {"score": 0.0, "generated_questions": []}

    user_prompt = f"Answer: {answer}"
    result = _call_judge_json(judge_client, RELEVANCY_SYSTEM_PROMPT, user_prompt)
    if (
        not isinstance(result, dict)
        or "questions" not in result
        or not isinstance(result["questions"], list)
    ):
        log.warning("Relevancy judge returned unexpected shape: %r", result)
        return None

    generated = [q for q in result["questions"] if isinstance(q, str) and q.strip()][:n_questions]
    if not generated:
        return {"score": 0.0, "generated_questions": []}

    import numpy as np

    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    gen_embs = embedder.encode(generated, normalize_embeddings=True)

    similarities = [float(np.dot(query_emb, g)) for g in gen_embs]
    return {"score": sum(similarities) / len(similarities), "generated_questions": generated}


def judge_correctness(answer: str, reference_answer: str, judge_client) -> dict | None:
    """Returns {"score": float 0-1 (F1), "precision": float, "recall": float,
    "tp": int, "fp": int, "fn": int}, or None if judging failed. Claim-overlap
    F1 between the candidate answer and a reference answer -- the factual
    core of RAGAS's answer_correctness metric, computed deterministically
    from the judge's claim extraction."""
    user_prompt = f"Reference answer: {reference_answer}\n\nCandidate answer: {answer}"
    result = _call_judge_json(judge_client, CORRECTNESS_SYSTEM_PROMPT, user_prompt)
    if not isinstance(result, dict) or "reference_claims" not in result:
        log.warning("Correctness judge returned unexpected shape: %r", result)
        return None

    ref_claims = result.get("reference_claims", [])
    extra_claims = result.get("candidate_extra_claims", [])

    tp = sum(1 for c in ref_claims if c.get("present_in_candidate") is True)
    fn = sum(1 for c in ref_claims if c.get("present_in_candidate") is False)
    fp = len(extra_claims)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "score": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_queries(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f).get("queries", [])


def load_references(path: Path) -> dict[str, dict]:
    if not path.exists():
        log.warning(
            "%s not found -- answer_correctness will be skipped for all queries. "
            "Run eval/generate_references.py first to enable it.",
            path,
        )
        return {}
    with open(path) as f:
        return yaml.safe_load(f).get("references", {})


def run_eval(
    query_specs: list[dict],
    ctx: AgentContext,
    agent_client,
    judge_client,
    agent_model: str,
    embedder,
    references: dict[str, dict] | None = None,
    sleep_between_judge_calls: float = 4.5,
    sleep_between_queries: float = 3.0,
) -> list[dict]:
    references = references or {}
    results = []
    for i, spec in enumerate(query_specs):
        query = spec["query"]
        log.info("Running agent for: %s", query)
        agent_result = run_agent(query, ctx, agent_client, agent_model)

        dollar_cost, cost_source = estimate_dollar_cost(
            agent_result["tokens"]["input_estimate"],
            agent_result["tokens"]["output_estimate"],
            agent_model,
        )

        row = {
            "query": query,
            "type": spec.get("type", "clean"),
            "has_answer": agent_result["has_answer"],
            "answer": agent_result["answer"],
            "mechanically_grounded": agent_result["grounded"],
            "all_citations_verified": agent_result["all_citations_verified"],
            "num_citations": len(agent_result["citations"]),
            "infra_failure": agent_result.get("infra_failure", False),  # real gap fixed here --
            # a transient connection error to the LLM provider produces the SAME
            # has_answer=False shape as a genuine model decline (short, generic
            # fallback text, no tool calls, no real citations), and without this
            # flag every infra failure was silently counted as real model
            # behavior in the "Declined to answer" aggregate. Observed in
            # practice: a proxy connection error cascaded into 7 consecutive
            # "declines" that were actually never-really-attempted queries.
            "faithfulness": None,
            "relevancy": None,
            "correctness": None,
            "tokens_total": agent_result["tokens"]["total_estimate"],
            "latency_s": agent_result["latency"]["total_s"],
            "dollar_cost_estimate": dollar_cost,
            "dollar_cost_source": cost_source,
        }

        if agent_result["has_answer"] and agent_result["citations"]:
            cited_texts = [
                ctx.corpus_by_id[c["chunk_id"]]["text"]
                for c in agent_result["citations"]
                if c.get("verified") and c["chunk_id"] in ctx.corpus_by_id
            ]

            try:
                row["faithfulness"] = judge_faithfulness(
                    query, agent_result["answer"], cited_texts, judge_client
                )
                time.sleep(sleep_between_judge_calls)

                row["relevancy"] = judge_relevancy(
                    query, agent_result["answer"], judge_client, embedder
                )
                time.sleep(sleep_between_judge_calls)

                ref = references.get(query)
                if ref:
                    row["correctness"] = judge_correctness(
                        agent_result["answer"], ref["reference_answer"], judge_client
                    )
                    time.sleep(sleep_between_judge_calls)
            except Exception as e:
                if getattr(e, "is_daily_limit", False):
                    # A daily quota is guaranteed to recur identically on every
                    # subsequent judge call until it resets (hours, not
                    # seconds) -- stop the whole run rather than burning
                    # through the remaining queries hitting the same wall.
                    log.error(
                        "Judge daily quota exhausted -- stopping the eval run early rather "
                        "than continuing to fail identically on remaining queries. This "
                        "query's row is saved with whatever metrics completed before the "
                        "limit hit."
                    )
                    results.append(row)
                    return results
                raise
        else:
            log.info("Skipping judging for '%s' -- no answer or no citations to check", query)

        results.append(row)

        if i < len(query_specs) - 1:
            time.sleep(sleep_between_queries)

    return results


def print_report(results: list[dict]):
    print("\n" + "=" * 70)
    print("PER-QUERY RESULTS")
    print("=" * 70)
    for r in results:
        print(f"[{r['type']}] {r['query']}")
        if r["faithfulness"] is not None:
            f = r["faithfulness"]
            print(
                f"    faithfulness: {f['score']:.2f} ({f['num_supported']}/{f['num_claims']} claims supported)"
            )
        if r["relevancy"] is not None:
            rel = r["relevancy"]
            print(f"    relevancy:    {rel['score']:.2f}")
        if r["correctness"] is not None:
            c = r["correctness"]
            print(
                f"    correctness:  {c['score']:.2f} (F1, precision={c['precision']:.2f} recall={c['recall']:.2f})"
            )
        if r["faithfulness"] is None and r["infra_failure"]:
            print(
                "    -> [INFRA FAILURE -- not a real decline] connection/provider error, query never really attempted"
            )
        elif r["faithfulness"] is None and not r["has_answer"]:
            print("    -> declined to answer (has_answer=False)")
        elif r["faithfulness"] is None:
            print("    -> answered but not judged (no verified citations to check)")
        cost_str = (
            f"${r['dollar_cost_estimate']:.5f} ({r['dollar_cost_source']})"
            if r["dollar_cost_estimate"] is not None
            else f"unknown ({r['dollar_cost_source']})"
        )
        print(f"    tokens~{r['tokens_total']}  latency={r['latency_s']:.2f}s  cost~{cost_str}")
        print()

    judged_f = [r["faithfulness"]["score"] for r in results if r["faithfulness"] is not None]
    judged_r = [r["relevancy"]["score"] for r in results if r["relevancy"] is not None]
    judged_c = [r["correctness"]["score"] for r in results if r["correctness"] is not None]
    infra_failures = [r for r in results if r["infra_failure"]]
    genuine_declines = [r for r in results if not r["has_answer"] and not r["infra_failure"]]

    print("=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)
    if infra_failures:
        print(
            f"NOTE: {len(infra_failures)}/{len(results)} scenario(s) hit an infrastructure/connection "
            f"failure (not a real model decline) and are EXCLUDED from the declined-to-answer count "
            f"below -- re-run those specific queries once the connection issue clears rather than "
            f"trusting this run's numbers as a complete picture."
        )
    print(
        f"1. Faithfulness (avg):  {sum(judged_f) / len(judged_f):.3f}  (n={len(judged_f)})"
        if judged_f
        else "1. Faithfulness (avg):  N/A"
    )
    print(
        f"2. Relevancy (avg):     {sum(judged_r) / len(judged_r):.3f}  (n={len(judged_r)})"
        if judged_r
        else "2. Relevancy (avg):     N/A"
    )
    print(
        f"3. Correctness (avg):   {sum(judged_c) / len(judged_c):.3f}  (n={len(judged_c)}, requires a reference answer)"
        if judged_c
        else "3. Correctness (avg):   N/A (no reference answers loaded -- run generate_references.py)"
    )

    real_rows = [r for r in results if not r["infra_failure"]]
    with_cost = [r for r in real_rows if r["dollar_cost_estimate"] is not None]
    if with_cost:
        avg_cost = sum(r["dollar_cost_estimate"] for r in with_cost) / len(with_cost)
        avg_tokens = sum(r["tokens_total"] for r in real_rows) / len(real_rows)
        avg_latency = sum(r["latency_s"] for r in real_rows) / len(real_rows)
        print(
            f"4. Cost per task (avg): ~{avg_tokens:.0f} tokens, {avg_latency:.2f}s, ~${avg_cost:.5f} "
            f"(n={len(real_rows)}, excludes infra failures -- their artificially tiny token counts "
            f"would skew this misleadingly low)"
        )

    print(f"\nDeclined to answer (genuine): {len(genuine_declines)}/{len(results)}")
    if infra_failures:
        print(f"Infra/connection failures (excluded above): {len(infra_failures)}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agent answer quality: faithfulness, relevancy, correctness"
    )
    parser.add_argument("--queries", type=Path, default=Path("eval/queries.yaml"))
    parser.add_argument("--references", type=Path, default=Path("eval/reference_answers.yaml"))
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-chunks"),
    )
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--episodes-config", type=Path, default=Path("config/episodes.yaml"))
    parser.add_argument("--provider", type=str, choices=["groq", "litellm"], default="groq")
    parser.add_argument(
        "--model", type=str, default=None, help="Overrides the provider's default model."
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        choices=["groq", "litellm", "gemini"],
        default="litellm",
        help="Which provider hosts the judge model. Default litellm -- Gemini's free tier "
        "turned out to cap at 20 requests/DAY, nowhere near enough for the claim-level "
        "judge (up to 3 calls/query), which is why this changed from the original default.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="minimax.minimax-m2.5",
        help="Cheap, tool-calling-capable model used as the judge (irrelevant that it's "
        "tool-calling-capable here specifically, just a convenient known-cheap option). "
        "Deliberately different from the answerer model by default to avoid "
        "self-preference bias -- a model judging its own answers tends to rate them "
        "more favorably. If --judge-provider gemini, defaults instead apply from "
        "get_gemini_client's own default handling.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        settings = yaml.safe_load(f)

    model = args.model or PROVIDER_DEFAULT_MODELS[args.provider]

    if args.judge_provider == args.provider and args.judge_model == model:
        log.warning(
            "Judge provider/model is IDENTICAL to the answerer provider/model (%s/%s) -- "
            "this risks self-preference bias (a model tends to rate its own answers more "
            "favorably). Consider a different --judge-provider/--judge-model unless this "
            "is deliberate.",
            args.provider,
            model,
        )

    query_specs = load_queries(args.queries)
    if not query_specs:
        log.error("No queries found in %s", args.queries)
        sys.exit(1)

    references = load_references(args.references)

    ctx = AgentContext(settings, args.chunks, args.episodes_config)
    agent_client = get_client_for_provider(args.provider)
    judge_client = get_judge_client(args.judge_provider, args.judge_model)
    embedder = get_embedder(settings)

    results = run_eval(query_specs, ctx, agent_client, judge_client, model, embedder, references)
    print_report(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "provider": args.provider,
                    "model": model,
                    "judge_provider": args.judge_provider,
                    "judge_model": args.judge_model,
                    "results": results,
                },
                f,
                indent=2,
            )
        log.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
