"""
Agent trajectory eval: runs a fixed scenario set (eval/agent_scenarios.yaml)
through the REAL agent and scores the captured trajectory (not just the
final answer) against structural expectations -- mechanically, no LLM
judge, matching the pattern already used for citation verification
elsewhere in this project.

Five metrics, as specified:

- task success rate: did the agent reach the correct final OUTCOME --
  correctly answering scenarios with a real answer in the corpus, and
  correctly declining ones that don't have one. Scenarios with
  expects_has_answer=null are tracked but not counted toward this rate,
  since there's no single correct outcome to score against.

- trajectory efficiency: how many tool calls it took vs. a scenario-defined
  reasonable maximum. efficiency = min(1.0, max_reasonable_tool_calls /
  actual_tool_calls) per scenario, averaged. A score under 1.0 means the
  agent needed more calls than expected to reach its outcome (whatever that
  outcome was) -- not by itself a failure, but a signal worth watching.

- tool selection accuracy: for scenarios with a structural expectation
  (list_episodes before search, get_episode_info instead of search,
  multiple distinct episodes searched for cross-episode questions), did the
  trajectory actually satisfy it. Binary per scenario, averaged.

- error recovery rate: of scenarios where at least one tool call in the
  trajectory returned an error (invalid episode_id, bad arguments, etc.),
  what fraction still reached task success afterward. Only computed over
  scenarios that actually hit an error -- an agent that never errors has an
  undefined (not 0%, not 100%) recovery rate, reported as N/A.

- cost per task: average token estimate and latency per scenario, plus a
  notional dollar cost using the SAME per-token equivalent rates Groq and
  Google publish for their paid tiers -- reported as a "what production
  would cost" estimate even though this project runs on free tiers, since
  Section 15.2 of the architecture doc already treats cost-per-query as a
  first-class production metric this project didn't previously compute
  anywhere.

Usage:
    python eval/eval_agent.py
    python eval/eval_agent.py --output eval/results/agent_trajectory.json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_agent")

# Fallback rates, used ONLY when a model isn't found in litellm's cost
# database (e.g. Groq's own models aren't in there, since Groq isn't a
# LiteLLM-routed provider in this project). NOT what this project actually
# pays on Groq's free tier -- a stand-in for "what would this cost at
# production/paid-tier rates," matching the framing already used in the
# architecture doc's production-scale sections.
GROQ_LLAMA_3_3_70B_PER_1M_INPUT = 0.59
GROQ_LLAMA_3_3_70B_PER_1M_OUTPUT = 0.79


def estimate_dollar_cost(tokens_in: int, tokens_out: int, model: str) -> tuple[float | None, str]:
    """Returns (cost, source). Prefers litellm's built-in cost database
    (litellm.model_cost) -- it's self-maintaining and already proven
    accurate against this project's actual proxy pricing (spot-checked
    minimax.minimax-m2.5 and moonshotai.kimi-k2.5 against the real
    per-model rates from the proxy's own config, both matched exactly).
    This matters because the earlier hardcoded-Groq-rate version was
    silently WRONG for every non-Groq run -- it reported Groq's price
    applied to whatever model actually answered, which is meaningless once
    this project started running against other providers via LiteLLM.

    Falls back to the Groq constants only for models litellm's database
    doesn't recognize (Groq's own models aren't LiteLLM-routed in this
    project, so they're never in that table) -- and returns "unknown" (cost
    None) rather than a silently wrong number if even that fallback isn't
    a reasonable match, so callers can distinguish "no cost data" from "$0"."""
    try:
        import litellm

        entry = litellm.model_cost.get(model)
        if entry and "input_cost_per_token" in entry and "output_cost_per_token" in entry:
            cost = (
                tokens_in * entry["input_cost_per_token"]
                + tokens_out * entry["output_cost_per_token"]
            )
            return cost, "litellm"
    except ImportError:
        pass

    if "llama-3.3-70b" in model:
        cost = (tokens_in / 1_000_000) * GROQ_LLAMA_3_3_70B_PER_1M_INPUT + (
            tokens_out / 1_000_000
        ) * GROQ_LLAMA_3_3_70B_PER_1M_OUTPUT
        return cost, "groq_fallback_estimate"

    return None, "unknown"


def load_scenarios(path: Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f).get("scenarios", [])


def check_tool_selection(scenario: dict, trajectory: list[dict]) -> tuple[bool | None, str]:
    """Returns (passed, reason). passed=None means this scenario has no
    tool-selection expectation to check (not every scenario type does)."""
    tool_names = [t["tool"] for t in trajectory]

    if scenario.get("requires_list_episodes_before_search"):
        first_search_idx = next(
            (i for i, t in enumerate(trajectory) if t["tool"] == "search_transcripts"), None
        )
        list_episodes_idx = next(
            (i for i, t in enumerate(trajectory) if t["tool"] == "list_episodes"), None
        )
        if first_search_idx is None:
            return None, "no search_transcripts call made -- nothing to check ordering against"
        if list_episodes_idx is None:
            return False, "search_transcripts called without ever calling list_episodes first"
        if list_episodes_idx > first_search_idx:
            return False, "search_transcripts called BEFORE list_episodes"

        expected_guest = scenario.get("expected_guest")
        if expected_guest:
            guest_arg = (trajectory[list_episodes_idx]["args"].get("guest") or "").lower()
            if expected_guest.lower() not in guest_arg and guest_arg not in expected_guest.lower():
                return (
                    False,
                    f"list_episodes called with guest={guest_arg!r}, expected something matching {expected_guest!r}",
                )

        return True, "list_episodes correctly preceded search_transcripts"

    if scenario.get("requires_get_episode_info"):
        if "get_episode_info" not in tool_names:
            return False, "expected get_episode_info, never called"
        return True, "get_episode_info was called as expected"

    if scenario.get("requires_multiple_episodes"):
        episode_ids = {
            t["args"].get("episode_id")
            for t in trajectory
            if t["tool"] == "search_transcripts" and t["args"].get("episode_id")
        }
        if len(episode_ids) < 2:
            return (
                False,
                f"expected search across >=2 episodes, only searched {len(episode_ids)}: {episode_ids}",
            )
        return True, f"searched across {len(episode_ids)} distinct episodes"

    return None, "no tool-selection expectation defined for this scenario"


def check_task_success(scenario: dict, result: dict) -> bool | None:
    """Returns None if this scenario has no defined correct outcome
    (expects_has_answer is null/absent), OR if the result reflects an
    infrastructure failure (daily quota exhausted, retries exhausted)
    rather than a genuine model outcome -- either way, there's nothing
    meaningful to score. Conflating an infra failure with "the agent
    failed to answer" was a real bug: it silently deflates task success
    rate with failures that have nothing to do with the agent's actual
    reasoning, exactly what happened when a Groq daily token limit was
    hit mid-eval-run and every subsequent scenario got counted as a
    genuine failure."""
    if result.get("infra_failure"):
        return None

    expected = scenario.get("expects_has_answer")
    if expected is None:
        return None

    if expected is True:
        return bool(result["has_answer"]) and bool(result["all_citations_verified"])
    else:
        return not result["has_answer"]


def run_scenario(scenario: dict, ctx: AgentContext, client, model: str) -> dict:
    result = run_agent(scenario["query"], ctx, client, model)
    trajectory = result["trajectory"]

    tool_call_count = len([t for t in trajectory if t["tool"] != "submit_answer"])
    had_error = any(t.get("error") for t in trajectory)
    task_success = check_task_success(scenario, result)

    if result.get("infra_failure"):
        # Never got a fair chance to run -- scoring tool selection here would
        # misleadingly show e.g. "expected get_episode_info, never called" for
        # a scenario that never got past the first (failed) LLM call at all.
        tool_selection_ok, tool_selection_reason = (
            None,
            "skipped: infra failure, scenario did not get a fair run",
        )
    else:
        tool_selection_ok, tool_selection_reason = check_tool_selection(scenario, trajectory)

    max_calls = scenario.get("max_reasonable_tool_calls")
    efficiency = min(1.0, max_calls / tool_call_count) if max_calls and tool_call_count else None

    dollar_cost, cost_source = estimate_dollar_cost(
        result["tokens"]["input_estimate"], result["tokens"]["output_estimate"], model
    )

    return {
        "query": scenario["query"],
        "type": scenario["type"],
        "has_answer": result["has_answer"],
        "answer": result["answer"],
        "citations": result["citations"],
        "withheld_answer": result.get("withheld_answer"),
        "all_citations_verified": result["all_citations_verified"],
        "infra_failure": result.get("infra_failure", False),
        "infra_failure_reason": result.get("infra_failure_reason"),
        "task_success": task_success,
        "tool_call_count": tool_call_count,
        "trajectory": trajectory,
        "had_error": had_error,
        "error_then_recovered": had_error and bool(task_success),
        "tool_selection_ok": tool_selection_ok,
        "tool_selection_reason": tool_selection_reason,
        "efficiency": efficiency,
        "tokens_total": result["tokens"]["total_estimate"],
        "latency_s": result["latency"]["total_s"],
        "dollar_cost_estimate": dollar_cost,
        "dollar_cost_source": cost_source,
    }


def print_report(rows: list[dict]):
    print("\n" + "=" * 70)
    print("PER-SCENARIO RESULTS")
    print("=" * 70)
    for r in rows:
        success_str = (
            "N/A" if r["task_success"] is None else ("PASS" if r["task_success"] else "FAIL")
        )
        selection_str = (
            "N/A"
            if r["tool_selection_ok"] is None
            else ("OK" if r["tool_selection_ok"] else "WRONG")
        )
        eff_str = f"{r['efficiency']:.2f}" if r["efficiency"] is not None else "N/A"
        infra_tag = " [INFRA FAILURE -- excluded from scoring]" if r["infra_failure"] else ""
        print(f"[{r['type']}] {r['query']}{infra_tag}")
        print(
            f"    success={success_str}  tool_selection={selection_str} ({r['tool_selection_reason']})  "
            f"calls={r['tool_call_count']}  efficiency={eff_str}  error={r['had_error']}"
        )
        cost_str = (
            f"${r['dollar_cost_estimate']:.5f} ({r['dollar_cost_source']})"
            if r["dollar_cost_estimate"] is not None
            else f"unknown ({r['dollar_cost_source']})"
        )
        print(f"    tokens~{r['tokens_total']}  latency={r['latency_s']:.2f}s  cost~{cost_str}")

        # For failures specifically, show what the model actually said --
        # otherwise the report tells you THAT something failed but not WHY,
        # forcing a re-run or a manual JSON dig every time (a real gap fixed
        # here after exactly that happened during a real debugging session).
        if r["task_success"] is False:
            answer_snippet = (r["answer"] or "")[:200]
            print(f"    answer: {answer_snippet}{'...' if len(r['answer'] or '') > 200 else ''}")
            if r.get("withheld_answer"):
                withheld_snippet = r["withheld_answer"][:200]
                print(
                    f"    [withheld -- model's claim before zero-citation override]: {withheld_snippet}"
                )
            if r["citations"]:
                for c in r["citations"]:
                    flag = "" if c.get("verified") else " [UNVERIFIED]"
                    print(f"      cited: {c.get('chunk_id')}{flag}")
        print()

    # --- Metric 1: task success rate ---
    infra_failures = [r for r in rows if r["infra_failure"]]
    scored = [r for r in rows if r["task_success"] is not None]
    print("=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)
    if infra_failures:
        print(
            f"NOTE: {len(infra_failures)}/{len(rows)} scenario(s) hit an infrastructure failure "
            f"(daily quota / exhausted retries) and were EXCLUDED from all scoring below -- "
            f"re-run those specific queries once the quota resets rather than trusting these "
            f"numbers as a complete picture."
        )
    if scored:
        success_rate = sum(r["task_success"] for r in scored) / len(scored)
        print(
            f"1. Task success rate:        {success_rate:.1%}  (n={len(scored)}, scenarios with a defined correct outcome)"
        )
    else:
        print("1. Task success rate:        N/A (no scenarios with a defined outcome)")

    # --- Metric 2: trajectory efficiency ---
    with_efficiency = [r for r in rows if r["efficiency"] is not None]
    if with_efficiency:
        avg_efficiency = sum(r["efficiency"] for r in with_efficiency) / len(with_efficiency)
        print(
            f"2. Trajectory efficiency:    {avg_efficiency:.2f}  (1.0 = at or under the reasonable call budget)"
        )

    # --- Metric 3: tool selection accuracy ---
    with_selection = [r for r in rows if r["tool_selection_ok"] is not None]
    if with_selection:
        selection_rate = sum(r["tool_selection_ok"] for r in with_selection) / len(with_selection)
        print(
            f"3. Tool selection accuracy:  {selection_rate:.1%}  (n={len(with_selection)}, scenarios with a checkable expectation)"
        )

    # --- Metric 4: error recovery rate ---
    with_errors = [r for r in rows if r["had_error"]]
    if with_errors:
        recovery_rate = sum(r["error_then_recovered"] for r in with_errors) / len(with_errors)
        print(
            f"4. Error recovery rate:      {recovery_rate:.1%}  (n={len(with_errors)}, scenarios that hit >=1 tool error)"
        )
    else:
        print("4. Error recovery rate:      N/A (no scenarios hit an error this run)")

    # --- Metric 5: cost per task ---
    avg_tokens = sum(r["tokens_total"] for r in rows) / len(rows)
    avg_latency = sum(r["latency_s"] for r in rows) / len(rows)
    with_cost = [r for r in rows if r["dollar_cost_estimate"] is not None]
    if with_cost:
        avg_cost = sum(r["dollar_cost_estimate"] for r in with_cost) / len(with_cost)
        cost_sources = {r["dollar_cost_source"] for r in with_cost}
        unknown_note = (
            f", {len(rows) - len(with_cost)} scenario(s) had no cost data"
            if len(with_cost) < len(rows)
            else ""
        )
        print(
            f"5. Cost per task (avg):      ~{avg_tokens:.0f} tokens, {avg_latency:.2f}s, "
            f"~${avg_cost:.5f} (source: {', '.join(cost_sources)}{unknown_note})"
        )
    else:
        print(
            f"5. Cost per task (avg):      ~{avg_tokens:.0f} tokens, {avg_latency:.2f}s, cost unknown "
            f"(model not found in litellm's cost database and no fallback rate matched)"
        )

    print(f"\nTotal scenarios run: {len(rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agent trajectories against structural expectations"
    )
    parser.add_argument("--scenarios", type=Path, default=Path("eval/agent_scenarios.yaml"))
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-chunks"),
    )
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--episodes-config", type=Path, default=Path("config/episodes.yaml"))
    parser.add_argument(
        "--provider",
        type=str,
        choices=["groq", "litellm"],
        default="groq",
        help="groq (default, free tier) or litellm (billed per-token -- 27 scenarios x "
        "several tool calls each adds up, budget accordingly).",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Overrides the provider's default model."
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--sleep-between-scenarios",
        type=float,
        default=5.0,
        help="Seconds to wait between scenarios. Each scenario runs the full agent loop "
        "(up to MAX_ITERATIONS Groq calls), so 27 scenarios back-to-back with no pacing "
        "can mean 100+ calls total -- this is the same TPM-not-RPM lesson learned "
        "elsewhere in this project (sponsor_extractor.py, ad_classifier.py), just applied "
        "here since each 'unit' of work is a whole agent run, not a single API call.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        settings = yaml.safe_load(f)

    scenarios = load_scenarios(args.scenarios)
    if not scenarios:
        log.error("No scenarios found in %s", args.scenarios)
        sys.exit(1)

    model = args.model or PROVIDER_DEFAULT_MODELS[args.provider]
    if args.provider == "litellm":
        log.warning(
            "Using litellm provider (%s) -- this is billed per-token, not free. "
            "%d scenarios x several tool calls each will incur real cost.",
            model,
            len(scenarios),
        )

    ctx = AgentContext(settings, args.chunks, args.episodes_config)
    client = get_client_for_provider(args.provider)

    rows = []
    for i, scenario in enumerate(scenarios):
        log.info("[%d/%d] Running: %s", i + 1, len(scenarios), scenario["query"])
        row = run_scenario(scenario, ctx, client, model)
        rows.append(row)

        if row.get("infra_failure_reason") == "daily_limit":
            log.error(
                "Daily token quota exhausted -- stopping early rather than burning through "
                "the remaining %d scenario(s), which would fail identically until the quota "
                "resets. Re-run with --scenarios pointed at a file containing just the "
                "remaining queries once the quota clears.",
                len(scenarios) - i - 1,
            )
            break

        if i < len(scenarios) - 1:
            time.sleep(args.sleep_between_scenarios)

    print_report(rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"timestamp": datetime.now(UTC).isoformat(), "results": rows}, f, indent=2)
        log.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
