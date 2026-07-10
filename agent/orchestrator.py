"""
Agent orchestrator: a tool-calling loop over Groq's chat completions API
(OpenAI-compatible), giving the model access to search_transcripts,
list_episodes, and get_episode_info (agent/tools.py), plus a final
submit_answer tool the model must call to finish. The model decides which
tool(s) to call and in what order -- single-episode lookup vs. cross-episode
synthesis is a routing decision made by the LLM itself based on the system
prompt and the tools available, not a separate hand-coded classifier.

Grounding: submit_answer requires structured citations (episode_id + start +
end for every claim). Every citation is mechanically checked against chunks
the model was actually shown this session -- a citation that doesn't match a
real retrieved chunk is a caught hallucination, not a trusted answer. This
is a cheap, code-level check, not another LLM call. It cannot catch subtler
misrepresentation of correctly-cited content -- that needs a faithfulness
eval pass (see eval/eval_answers.py, planned), which this does not replace.

Context management: within a session, tool results are deduplicated by
chunk_id (the same chunk retrieved twice via different queries is only
counted/sent once) and the total number of distinct chunks accumulated is
capped (MAX_SESSION_CHUNKS) -- a bounded budget regardless of how many
search calls the model makes across MAX_ITERATIONS. Token usage is tracked
per call via a rough estimate (chars/4), logged and returned, not enforced
against a hard ceiling -- Llama 3.3's context window is large enough that
overflow is not a realistic risk at this corpus size, so this is cost/
diagnostic visibility, not overflow prevention. If this pipeline is ever
pointed at a much larger corpus or a smaller-context model, the tracked
numbers are what tells you it's time to add real enforcement.

Latency: wall-clock time is recorded per LLM call and per tool call
separately, so a slow query's breakdown (stuck waiting on the model vs.
stuck in retrieval) is visible rather than only a total.

Loop shape: send messages + tool schemas -> if the model responds with
tool_calls, execute each locally (including submit_answer, which ends the
loop), append results as tool-role messages, and call again -> repeat until
submit_answer is called or MAX_ITERATIONS is hit (a runaway-loop guard).

Usage:
    python agent/orchestrator.py --query "What does Kaldellis say about Justinian?"
    python agent/orchestrator.py --query "..." --verbose
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.tools import (  # noqa: E402
    TOOL_SCHEMAS,
    get_episode_info_tool,
    list_episodes_tool,
    search_transcripts_tool,
)
from retrieve.hybrid import build_bm25, get_collection, get_embedder, load_corpus  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("orchestrator")

SYSTEM_PROMPT = """You are a research assistant that answers questions about the \
Lex Fridman Podcast using only the tools provided -- you have no other knowledge \
of episode content.

Routing guidance:
- If the question names a specific guest or episode, use list_episodes first to \
find the episode_id, then use search_transcripts restricted to that episode_id. \
Never guess or construct an episode_id yourself.
- If the question spans multiple guests or episodes, call search_transcripts \
multiple times (once per relevant episode, or unrestricted) and synthesize \
across the results.
- If the question is about an episode itself (who was the guest, when did it \
air) rather than its content, use get_episode_info instead of searching.
- If any tool call returns an error, that is not a dead end -- read the error, \
fix the problem it describes (e.g. call list_episodes to get a real episode_id), \
and try again. Do not abandon the tool and answer from memory instead just \
because one call failed.

Grounding rules -- these are strict, not suggestions:
- You must finish by calling submit_answer. Do not answer in plain text.
- Every citation in submit_answer must reference a chunk_id, episode_id, start, \
and end that a search_transcripts call actually returned to you this turn. \
Do not cite anything you were not shown.
- If a search_transcripts call returns no_relevant_results=true, or the \
available results don't actually support an answer, set has_answer=false and \
explain what you found instead -- do not guess, and do not answer from \
general knowledge about the guest or topic.
- has_answer=true requires at least one real citation. An answer with no \
citations is not acceptable.
- When has_answer=false, the `answer` field must plainly state that you could \
not find or verify an answer. Do not write a confident-sounding explanation \
that reads like a real answer -- if you don't have verified evidence, say so \
directly and briefly."""

MAX_ITERATIONS = 6
MAX_SESSION_CHUNKS = 30  # cap on distinct chunks accumulated across the whole session
MAX_LLM_RETRIES = 3  # retries specifically for tool_use_failed -- the model emitted a
# malformed tool call (e.g. pseudo-XML instead of structured JSON);
# this is a generation-sampling issue, often transient, not
# something an invalid-schema bug this time -- worth a couple of
# retries before giving up, but not worth retrying indefinitely
# since a genuinely broken prompt/schema will just fail every time


def _is_tool_use_failed(error: Exception) -> bool:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        code = (body.get("error") or {}).get("code")
        if code == "tool_use_failed":
            return True
    return "tool_use_failed" in str(error)


# Some Llama checkpoints on Groq occasionally fall back to a "pythonic" tool
# call format instead of the structured JSON tool_calls Groq's parser
# expects. Observed in the wild in (at least) two slightly different shapes:
#   <function=list_episodes{"guest": "Kaldellis"}</function>
#   <function=search_transcripts>{"episode_id": "...", ...}</function>
# (note the second has a ">" between the name and the JSON args, the first
# doesn't) -- the ">" is made optional to match both. When this happens,
# Groq rejects the generation as a 400 but still returns the raw text via
# failed_generation, which is salvageable: this is a deterministic model
# habit for a given prompt, not sampling noise, so blindly retrying the same
# request tends to reproduce the identical malformed output rather than
# fixing itself. Parsing the salvageable text is more reliable than retrying.
_PSEUDO_FUNCTION_CALL_RE = re.compile(r"<function=(\w+)>?(\{.*?\})</function>", re.DOTALL)


def _salvage_tool_call(error: Exception):
    """Attempts to recover a usable tool call from a tool_use_failed error's
    failed_generation text. Returns a response-shaped SimpleNamespace
    matching what call sites expect from a normal API response, or None if
    nothing salvageable was found."""
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return None

    failed_generation = (body.get("error") or {}).get("failed_generation", "")
    match = _PSEUDO_FUNCTION_CALL_RE.search(failed_generation)
    if not match:
        return None

    name, args_str = match.group(1), match.group(2)
    try:
        json.loads(args_str)  # validate it's real JSON before trusting it downstream
    except json.JSONDecodeError:
        log.warning("Salvage regex matched but arguments were not valid JSON: %r", args_str)
        return None

    log.warning(
        "Salvaged a tool call from a tool_use_failed error: %s(%s) -- the model used a "
        "non-standard call format Groq's parser rejected, recovered it from the raw text "
        "instead of retrying the identical request",
        name,
        args_str,
    )

    from types import SimpleNamespace

    synthetic_tool_call = SimpleNamespace(
        id=f"salvaged_{uuid.uuid4().hex[:8]}",
        function=SimpleNamespace(name=name, arguments=args_str),
    )
    message = SimpleNamespace(content=None, tool_calls=[synthetic_tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def call_llm_with_retry(client, max_retries: int = MAX_LLM_RETRIES, **kwargs):
    """Wraps client.chat.completions.create. On tool_use_failed, first tries
    to salvage a usable tool call from the raw failed_generation text (see
    _salvage_tool_call) since this failure mode is usually a deterministic
    model habit, not transient noise -- retrying the identical request tends
    to reproduce the same malformed output. Falls back to retry-with-backoff
    if salvage isn't possible. Any other error (auth, rate limit, etc.) is
    raised immediately -- retrying those wouldn't help and would just mask a
    real problem."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if not _is_tool_use_failed(e):
                raise

            salvaged = _salvage_tool_call(e)
            if salvaged:
                return salvaged

            last_error = e
            if attempt < max_retries - 1:
                wait = 1.5**attempt
                log.warning(
                    "tool_use_failed on attempt %d/%d (unsalvageable) -- retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
    log.error("tool_use_failed persisted after %d attempts, giving up", max_retries)
    raise last_error


SUBMIT_ANSWER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit your final answer. This ends the conversation -- call this exactly once, when ready.",
        "parameters": {
            "type": "object",
            "properties": {
                "has_answer": {
                    "type": "boolean",
                    "description": "True if the tools returned enough relevant information to answer. False if not -- explain what you found instead in `answer`.",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer text, or an explanation of why the question can't be answered from the available content.",
                },
                "citations": {
                    "type": "array",
                    "description": "Every chunk actually used to support the answer. Must be empty if has_answer is false.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "episode_id": {"type": "string"},
                            "guest": {"type": "string"},
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                        },
                        "required": ["chunk_id", "episode_id", "start", "end"],
                    },
                },
            },
            "required": ["has_answer", "answer", "citations"],
        },
    },
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4) -- not a real tokenizer, good enough
    for cost/budget visibility rather than precise accounting. Swap for a
    real tokenizer (e.g. tiktoken, or Llama's own) if precise counts ever
    matter more than they do at this corpus/query scale."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(json.dumps(m))
    return total


@dataclass
class SessionState:
    """Per-query session state: dedup, chunk budget, token/latency tracking.
    Fresh per run_agent() call -- this is not cross-query memory."""

    seen_chunk_ids: set[str] = field(default_factory=set)
    retrieved_chunks: dict[str, dict] = field(default_factory=dict)  # chunk_id -> chunk data
    tokens_in: int = 0
    tokens_out: int = 0
    llm_call_latencies: list[float] = field(default_factory=list)
    tool_call_latencies: list[dict] = field(default_factory=list)  # [{"tool": name, "seconds": s}]
    iterations_used: int = 0

    @property
    def chunk_budget_remaining(self) -> int:
        return max(0, MAX_SESSION_CHUNKS - len(self.seen_chunk_ids))

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def total_latency_s(self) -> float:
        return sum(self.llm_call_latencies) + sum(t["seconds"] for t in self.tool_call_latencies)


class AgentContext:
    """Holds the loaded corpus/index/models once, reused across tool calls
    within and across turns -- avoids reloading the embedder or rebuilding
    the BM25 index on every single tool invocation."""

    def __init__(self, settings: dict, chunks_dir: Path, episodes_config: Path):
        self.settings = settings
        log.info("Loading corpus from %s", chunks_dir)
        self.corpus = load_corpus(chunks_dir)
        self.corpus_by_id = {r["chunk_id"]: r for r in self.corpus}
        log.info("Corpus loaded: %d chunks", len(self.corpus))

        log.info("Building BM25 index...")
        self.bm25 = build_bm25(self.corpus)

        log.info("Loading embedder and Chroma collection...")
        self.embedder = get_embedder(settings)
        self.collection = get_collection(settings)

        with open(episodes_config) as f:
            data = yaml.safe_load(f)
        self.episode_meta = {ep["id"]: ep for ep in data.get("episodes", [])}


def get_groq_client():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        from groq import Groq
    except ImportError:
        log.error("groq package not installed. Run: pip install groq")
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.error("GROQ_API_KEY not set.")
        sys.exit(1)

    return Groq(api_key=api_key)


def dedup_and_budget_search_results(result: dict, session: SessionState) -> dict:
    """Filters out chunks already seen this session and enforces the total
    session chunk cap. Applied only to search_transcripts results -- the
    other two tools don't return retrievable chunks."""
    if "results" not in result:
        return result

    remaining_budget = session.chunk_budget_remaining
    kept = []
    for chunk in result["results"]:
        cid = chunk["chunk_id"]
        if cid in session.seen_chunk_ids:
            continue  # already shown this session, skip re-sending it
        if len(kept) >= remaining_budget:
            break  # session chunk cap reached
        kept.append(chunk)
        session.seen_chunk_ids.add(cid)
        session.retrieved_chunks[cid] = chunk

    new_result = dict(result)
    new_result["results"] = kept
    if len(kept) < len(result["results"]):
        new_result["note"] = (
            new_result.get("note", "")
            + " Some results omitted: already seen or session chunk budget reached."
        ).strip()
    return new_result


def _build_schema_lookup() -> dict[str, dict]:
    """Maps tool name -> its JSON schema, for argument validation before dispatch."""
    lookup = {}
    for schema in [*TOOL_SCHEMAS, SUBMIT_ANSWER_SCHEMA]:
        fn = schema["function"]
        lookup[fn["name"]] = fn["parameters"]
    return lookup


_SCHEMA_LOOKUP = _build_schema_lookup()


def validate_tool_args(name: str, args: dict) -> str | None:
    """Validates args against the tool's JSON schema (type mismatches like a
    string where an integer is required -- the exact case that broke
    search_transcripts with top_k="5") before anything executes. Returns an
    error message if invalid, None if valid.

    This exists because a malformed argument either crashes downstream (if
    the type mismatch triggers an exception) or, worse, silently misbehaves
    -- e.g. top_k="5" as a string would make `top_k * 3` in
    search_transcripts_tool evaluate to string repetition ("555") instead of
    erroring, quietly corrupting the results rather than failing loudly.
    Rejecting bad input before dispatch turns both failure modes into one
    clear, actionable error the model can see and correct."""
    schema = _SCHEMA_LOOKUP.get(name)
    if not schema:
        return None  # unknown tool -- let the existing dispatch error handle it

    from jsonschema import Draft7Validator

    errors = sorted(Draft7Validator(schema).iter_errors(args), key=lambda e: e.path)
    if not errors:
        return None

    return "; ".join(f"{'/'.join(str(p) for p in e.path) or '(root)'}: {e.message}" for e in errors)


def validate_episode_id(episode_id: str | None, ctx: AgentContext) -> str | None:
    """Checks a caller-supplied episode_id actually exists before running a
    search restricted to it. Without this, a model that guesses/fabricates
    an episode_id (e.g. the literal string "Jensen Huang episode id" instead
    of calling list_episodes first) gets back an empty, unhelpful result
    with no signal about *why* -- it just looks like "no relevant content,"
    indistinguishable from a genuinely unanswerable question. An explicit
    error tells the model exactly what went wrong and how to fix it."""
    if episode_id is None:
        return None
    if episode_id not in ctx.episode_meta:
        return (
            f"episode_id {episode_id!r} does not exist. Call list_episodes first to "
            "find the correct episode_id -- do not guess or construct one."
        )
    return None


def execute_tool_call(
    name: str, args: dict, ctx: AgentContext, session: SessionState, verbose: bool
) -> dict:
    """Dispatches a single tool call by name. Returns a JSON-serializable
    result -- errors are returned as a normal result (an {"error": ...}
    dict), not raised, so the model sees the failure and can adjust its
    plan rather than the whole turn crashing.

    Validates arguments (schema types) and, for tools taking an episode_id,
    that it actually exists, before any real work happens -- see
    validate_tool_args and validate_episode_id for why each matters."""
    start = time.time()

    schema_error = validate_tool_args(name, args)
    if schema_error:
        log.warning("Tool %s called with invalid arguments: %s", name, schema_error)
        result = {"error": f"Invalid arguments: {schema_error}"}
        elapsed = time.time() - start
        session.tool_call_latencies.append({"tool": name, "seconds": round(elapsed, 3)})
        return result

    episode_id_error = (
        validate_episode_id(args.get("episode_id"), ctx)
        if name in ("search_transcripts", "get_episode_info")
        else None
    )
    if episode_id_error:
        log.warning("Tool %s called with invalid episode_id: %s", name, episode_id_error)
        result = {"error": episode_id_error}
        elapsed = time.time() - start
        session.tool_call_latencies.append({"tool": name, "seconds": round(elapsed, 3)})
        return result

    try:
        if name == "search_transcripts":
            result = search_transcripts_tool(
                query=args["query"],
                corpus=ctx.corpus,
                corpus_by_id=ctx.corpus_by_id,
                bm25=ctx.bm25,
                collection=ctx.collection,
                embedder=ctx.embedder,
                settings=ctx.settings,
                episode_id=args.get("episode_id"),
                top_k=args.get("top_k"),
            )
            result = dedup_and_budget_search_results(result, session)

        elif name == "list_episodes":
            result = {
                "results": list_episodes_tool(
                    ctx.episode_meta, guest=args.get("guest"), tag=args.get("tag")
                )
            }

        elif name == "get_episode_info":
            result = get_episode_info_tool(ctx.episode_meta, args["episode_id"])

        else:
            result = {"error": f"Unknown tool: {name}"}

    except Exception as e:
        log.warning("Tool %s failed: %s", name, e)
        result = {"error": str(e)}

    elapsed = time.time() - start
    session.tool_call_latencies.append({"tool": name, "seconds": round(elapsed, 3)})
    if verbose:
        log.info("Tool call: %s(%s) [%.2fs]", name, args, elapsed)

    return result


def verify_citations(citations: list[dict], session: SessionState) -> list[dict]:
    """Checks each citation against chunks actually retrieved this session.
    A citation is verified if its chunk_id was really shown to the model AND
    the episode_id/timestamp claimed matches what that chunk actually is --
    catching both fabricated chunk_ids and a chunk_id paired with mismatched
    metadata (a subtler, easy-to-miss form of the same problem).

    Also backfills guest (and any other display fields) from the actual
    retrieved chunk rather than trusting the model to have included them --
    `guest` is optional in the submit_answer schema, so relying on the model
    to remember it produces citations that display as "?" even when
    perfectly valid."""
    verified = []
    for c in citations:
        chunk = session.retrieved_chunks.get(c.get("chunk_id"))
        is_valid = (
            chunk is not None
            and chunk["episode_id"] == c.get("episode_id")
            and chunk["start"] == c.get("start")
            and chunk["end"] == c.get("end")
        )
        enriched = dict(c)
        if chunk:
            enriched["guest"] = chunk.get("guest", enriched.get("guest"))
        verified.append({**enriched, "verified": is_valid})
    return verified


def run_agent(query: str, ctx: AgentContext, client, model: str, verbose: bool = False) -> dict:
    session = SessionState()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    tools = TOOL_SCHEMAS + [SUBMIT_ANSWER_SCHEMA]

    for iteration in range(MAX_ITERATIONS):
        session.iterations_used = iteration + 1

        session.tokens_in += estimate_messages_tokens(messages)
        llm_start = time.time()
        try:
            response = call_llm_with_retry(
                client,
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            session.llm_call_latencies.append(round(time.time() - llm_start, 3))
            log.error("LLM call failed after retries: %s", e)
            return _finalize(
                has_answer=False,
                answer=(
                    "The model repeatedly failed to produce a valid tool call for this "
                    "question. This is usually transient -- try rephrasing the question "
                    "or running it again."
                ),
                citations=[],
                session=session,
                grounded=False,
            )
        session.llm_call_latencies.append(round(time.time() - llm_start, 3))

        message = response.choices[0].message
        session.tokens_out += estimate_tokens(message.content or "") + estimate_tokens(
            json.dumps([tc.function.arguments for tc in (message.tool_calls or [])])
        )

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            # Model skipped submit_answer and returned plain text -- accept it,
            # but flag it as ungrounded/unverified since it carries no citations.
            log.warning(
                "Model returned plain text instead of calling submit_answer -- no citations to verify"
            )
            return _finalize(
                has_answer=True,
                answer=message.content or "",
                citations=[],
                session=session,
                grounded=False,
            )

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            args = json.loads(tc.function.arguments)

            if tc.function.name == "submit_answer":
                verified_citations = verify_citations(args.get("citations", []), session)
                valid_citations = [c for c in verified_citations if c["verified"]]
                claimed_has_answer = args.get("has_answer", False)
                claimed_answer = args.get("answer", "")

                if claimed_has_answer and not valid_citations:
                    # The model said it has an answer, but not a single citation
                    # actually checks out against what it was really shown -- this
                    # is exactly the fabricated-episode_id failure mode: the model
                    # writes a plausible-sounding answer anyway despite having no
                    # real grounding. Don't pass that text through as if it were a
                    # real answer just because it reads confidently.
                    log.warning(
                        "Model claimed has_answer=True but zero citations verified -- "
                        "withholding the claimed answer as ungrounded"
                    )
                    return _finalize(
                        has_answer=False,
                        answer=(
                            "I wasn't able to find and verify a real answer to this "
                            "question in the podcast content -- what I found didn't "
                            "hold up when checked against the actual source material. "
                            "Try rephrasing, or this may not be covered in the corpus."
                        ),
                        citations=verified_citations,
                        session=session,
                        grounded=True,
                        withheld_answer=claimed_answer,
                    )

                return _finalize(
                    has_answer=claimed_has_answer,
                    answer=claimed_answer,
                    citations=verified_citations,
                    session=session,
                    grounded=True,
                )

            result = execute_tool_call(tc.function.name, args, ctx, session, verbose)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    log.warning("Hit MAX_ITERATIONS=%d without submit_answer being called", MAX_ITERATIONS)
    return _finalize(
        has_answer=False,
        answer="I wasn't able to fully resolve this question within the allowed number of steps.",
        citations=[],
        session=session,
        grounded=False,
    )


def _finalize(
    has_answer: bool,
    answer: str,
    citations: list[dict],
    session: SessionState,
    grounded: bool,
    withheld_answer: str | None = None,
) -> dict:
    unverified = [c for c in citations if not c.get("verified", True)]
    if unverified:
        log.warning(
            "%d citation(s) failed verification -- model cited chunks it wasn't shown",
            len(unverified),
        )

    return {
        "has_answer": has_answer,
        "answer": answer,
        "citations": citations,
        "grounded": grounded,  # False if the model bypassed submit_answer entirely
        "all_citations_verified": grounded and not unverified,
        "withheld_answer": withheld_answer,  # set if the model's claimed answer was
        # overridden because it had zero verified citations backing it -- kept here
        # for debugging/verbose output, never shown to the user as a real answer
        "iterations_used": session.iterations_used,
        "chunks_retrieved": len(session.seen_chunk_ids),
        "tokens": {
            "input_estimate": session.tokens_in,
            "output_estimate": session.tokens_out,
            "total_estimate": session.total_tokens,
        },
        "latency": {
            "total_s": round(session.total_latency_s, 3),
            "llm_s": round(sum(session.llm_call_latencies), 3),
            "tool_s": round(sum(t["seconds"] for t in session.tool_call_latencies), 3),
            "llm_calls": session.llm_call_latencies,
            "tool_calls": session.tool_call_latencies,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Query the podcast Q&A agent")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-chunks"),
    )
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--episodes-config", type=Path, default=Path("config/episodes.yaml"))
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        settings = yaml.safe_load(f)

    ctx = AgentContext(settings, args.chunks, args.episodes_config)
    client = get_groq_client()

    result = run_agent(args.query, ctx, client, args.model, verbose=args.verbose)

    if result["has_answer"]:
        print(f"\n{result['answer']}\n")
        if result["citations"]:
            print("Citations:")
            for c in result["citations"]:
                flag = (
                    "" if c.get("verified") else "  [UNVERIFIED -- not shown to model this session]"
                )
                print(
                    f"  - {c.get('guest', '?')} ({c.get('episode_id')}, {c.get('start')}-{c.get('end')}s){flag}"
                )
    else:
        # Deliberately visually distinct from a real answer -- has_answer=False
        # means citation verification found nothing to back this up, even if
        # the model's own wording sounds confident. Printing it identically to
        # a real answer would hide that distinction from the user, which is
        # exactly the failure mode all the grounding logic exists to prevent.
        print(f"\n[NO VERIFIED ANSWER] {result['answer']}\n")

    if args.verbose and result.get("withheld_answer"):
        print(
            f"[DEBUG] Model's original (withheld, ungrounded) answer was:\n  {result['withheld_answer']}\n"
        )

    print(
        f"[iterations={result['iterations_used']} chunks={result['chunks_retrieved']} "
        f"tokens~{result['tokens']['total_estimate']} "
        f"latency={result['latency']['total_s']}s "
        f"(llm={result['latency']['llm_s']}s, tools={result['latency']['tool_s']}s) "
        f"grounded={result['grounded']} all_citations_verified={result['all_citations_verified']}]"
    )


if __name__ == "__main__":
    main()
