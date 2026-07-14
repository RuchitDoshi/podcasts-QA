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
    build_episode_index,
    get_context_window_tool,
    get_episode_info_tool,
    list_episodes_tool,
    search_transcripts_tool,
)
from retrieve.hybrid import (  # noqa: E402
    build_bm25,
    get_collection,
    get_cross_encoder,
    get_embedder,
    load_corpus,
)

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
air) rather than its content, use get_episode_info instead of searching. When \
citing information from get_episode_info (not from a transcript search), use \
citation chunk_id="episode_info", the real episode_id, and start=0, end=0 -- \
this is a real citation type for episode-level metadata, not a chunk quote, \
and answers sourced this way still require this citation to count as grounded.
- If any tool call returns an error, that is not a dead end -- read the error, \
fix the problem it describes (e.g. call list_episodes to get a real episode_id), \
and try again. Do not abandon the tool and answer from memory instead just \
because one call failed.
- If a search_transcripts result seems cut off, references something not fully \
explained, or you need more surrounding context to answer accurately, call \
get_context_window with that result's chunk_id (or its prev_chunk_id/ \
next_chunk_id) rather than guessing at what the missing context might say.
- Do not call submit_answer in the same turn as other tool calls whose results \
you haven't seen yet. Send your exploratory/search calls, wait for their \
results, and only call submit_answer by itself once you've actually used them.
- Once you have enough well-supported material to give a clear, substantive \
answer, stop searching and call submit_answer -- do not keep issuing further \
searches chasing additional angles or more exhaustive coverage on a broad \
question. A good answer covering the main points with solid citations is the \
goal, not maximal coverage of every related sub-topic. If you're rephrasing \
the same underlying question multiple times without learning anything new, \
that's a sign to stop and answer with what you already have, not to keep going.

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


def _is_rate_limit_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status == 429:
        return True
    body = getattr(error, "body", None)
    if isinstance(body, dict) and (body.get("error") or {}).get("code") == "rate_limit_exceeded":
        return True
    msg = str(error).lower()
    return "rate limit" in msg or "429" in msg


def _is_daily_limit_error(error: Exception) -> bool:
    """Distinguishes a DAILY token quota exhaustion (TPD) from an ordinary
    per-minute rate limit (TPM/RPM). This distinction matters because the
    two need completely different responses: a per-minute limit clears in
    seconds and retrying with backoff works. A daily limit clears on the
    order of hours -- observed in practice: Groq's error gave a suggested
    wait of 7m32s, far longer than this project's total retry budget
    (~60s across all attempts), meaning every retry was guaranteed to fail
    before it even started. Retrying a daily limit the same way as a
    per-minute one just wastes time proving what the error message already
    said. Detected via the message text since Groq's error body doesn't
    expose a distinct machine-readable field for this."""
    msg = str(error).lower()
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        msg += " " + str((body.get("error") or {}).get("message", "")).lower()
    return "per day" in msg or "tpd" in msg or "daily" in msg


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
    """Wraps client.chat.completions.create with two independent retry paths:

    1. tool_use_failed -- first tries to salvage a usable tool call from the
       raw failed_generation text (see _salvage_tool_call), since this is
       usually a deterministic model habit, not sampling noise, and retrying
       the identical request tends to reproduce the same malformed output.
       Falls back to retry-with-backoff if salvage isn't possible.

    2. Plain per-minute rate limiting (429, TPM/RPM) -- retried with longer
       backoff. An earlier version of this function treated ANY
       non-tool_use_failed error as unretryable, including this one --
       which meant a real Groq rate limit (hit in practice running this
       project, not a hypothetical) crashed the whole agent run immediately
       instead of backing off.

    3. Daily token limit (429, TPD) -- NOT retried at all. A daily quota
       clears on the order of hours, not seconds -- observed in practice,
       Groq gave a suggested wait of 7m32s, far longer than this function's
       entire retry budget. Retrying anyway just burns time proving what
       the error already said. Raised immediately with is_daily_limit=True
       set on the exception, so callers (run_agent) can distinguish this
       from an ordinary exhausted-retries failure and avoid counting it as
       a genuine model/reasoning failure in eval scoring."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if _is_daily_limit_error(e):
                log.error(
                    "Daily token quota exhausted -- not retrying (would need to wait far "
                    "longer than this function's retry budget). See the error for Groq's "
                    "suggested wait time, or upgrade tier: %s",
                    e,
                )
                e.is_daily_limit = True
                raise

            if _is_rate_limit_error(e):
                last_error = e
                if attempt < max_retries - 1:
                    wait = 10.0 * (attempt + 1)  # rate limit windows reset on the order of
                    # seconds to a minute, not milliseconds -- longer backoff than tool_use_failed
                    log.warning(
                        "Rate limited, attempt %d/%d -- waiting %.0fs",
                        attempt + 1,
                        max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                break

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
    log.error("LLM call failed after %d attempts (last error: %s)", max_retries, last_error)
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
    trajectory: list[dict] = field(
        default_factory=list
    )  # [{"tool": name, "args": {...}, "error": bool}, ...] in call order --
    # the actual sequence of tool calls made this session, exposed for agent
    # trajectory eval (tool selection accuracy, error recovery, efficiency).
    # Distinct from tool_call_latencies, which only tracks timing.
    info_verified_episodes: set[str] = field(
        default_factory=set
    )  # episode_ids for which get_episode_info succeeded this session -- a
    # real gap found in practice: get_episode_info returns episode-level
    # metadata (air date, tags, description), not a chunk, so there was NO
    # valid citation shape for answers sourced from it. Every single
    # episode_info-type question failed task success as a result, 3/3 in one
    # eval run, despite the model getting genuinely correct data from the
    # tool every time -- this is what verify_citations checks against
    # instead of a chunk_id for that category of citation.

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
        self.episode_index = build_episode_index(self.corpus)
        log.info("Corpus loaded: %d chunks", len(self.corpus))

        log.info("Building BM25 index...")
        self.bm25 = build_bm25(self.corpus)

        log.info("Loading embedder and Chroma collection...")
        self.embedder = get_embedder(settings)
        self.collection = get_collection(settings)

        self.cross_encoder = None
        if settings["retrieval"].get("use_reranker", False):
            log.info("Loading cross-encoder reranker...")
            self.cross_encoder = get_cross_encoder(settings)

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


class _LangChainChatCompletionsAdapter:
    """Makes a LangChain ChatLiteLLM client expose the SAME
    client.chat.completions.create(model=, messages=, tools=, tool_choice=,
    temperature=) -> response.choices[0].message interface the rest of this
    file already expects from the Groq SDK. This means run_agent,
    execute_tool_call, _salvage_tool_call, and every retry/error-handling
    path work completely unchanged regardless of which underlying client is
    used -- only this adapter and get_litellm_client() know LangChain/LiteLLM
    exist.

    Why ChatLiteLLM specifically, not ChatOpenAI: two attempts preceded this
    one. First, reusing Groq's own SDK class pointed at a different base_url
    failed with a persistent 405 -- Groq's SDK makes Groq-specific
    assumptions that don't hold against an arbitrary target. Second,
    LangChain's ChatOpenAI (built on the `openai` package, assumes strict
    OpenAI wire-format compatibility) got further -- past the 405 -- but hit
    a server-side 'NoneType has no attribute startswith' error on every
    model tried, consistent across both plain and LangChain-constructed
    requests. A colleague who already has this exact proxy working
    confirmed they use ChatLiteLLM (from the `litellm` package, which is
    purpose-built to speak each provider's specific dialect through a
    LiteLLM proxy, not just generic OpenAI shape) -- this proxy apparently
    isn't a fully strict OpenAI-compatible target, which is exactly the gap
    ChatLiteLLM is designed to bridge.

    litellm.drop_params = True is set at import time below (module-level,
    once) rather than per-instance -- confirmed via the colleague's own code
    to be necessary for some providers behind this proxy (e.g. Bedrock
    rejects a `seed` param outright with UnsupportedParamsError); without
    this, a request carrying a param one specific backend model doesn't
    support would hard-fail instead of that param being silently dropped.

    custom_llm_provider="openai" (the default here) is necessary for a
    different reason found in practice: this proxy's model catalog
    (openai.gpt-oss-120b-1:0, us.anthropic.claude-..., qwen.qwen3-32b-v1:0,
    etc.) happens to match AWS Bedrock's own model-ID naming convention
    exactly. LiteLLM auto-detects provider from model-name PATTERNS, and
    for a Bedrock-shaped name it assumed native Bedrock routing and tried
    calling AWS directly via boto3/local AWS credentials -- completely
    bypassing api_base -- rather than forwarding the request to this
    proxy. Explicitly forcing custom_llm_provider="openai" disables that
    guesswork and makes LiteLLM treat api_base as the real target,
    regardless of what the model name happens to look like."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_tokens: int = 8000,
        custom_llm_provider: str = "openai",
    ):
        self._base_url = base_url
        self._api_key = api_key
        self._custom_llm_provider = custom_llm_provider
        self._max_tokens = max_tokens  # explicit cap -- the colleague's own code notes
        # that without this, structured/long-output requests intermittently
        # truncated mid-response on this proxy's default token ceiling
        self._llm_cache: dict[str, object] = {}  # keyed by model name

    def _get_llm(self, model: str, temperature: float):
        import litellm
        from langchain_litellm import ChatLiteLLM

        litellm.drop_params = True  # idempotent to set repeatedly; cheap, and
        # guarantees this is set even if this adapter is imported/used before
        # anything else in the process has had a chance to set it

        # Cache per (model, temperature) pair rather than rebuilding on every
        # call -- cheap either way at this call volume, but avoids repeated
        # client construction overhead within one session.
        key = f"{model}:{temperature}"
        if key not in self._llm_cache:
            self._llm_cache[key] = ChatLiteLLM(
                model=model,
                temperature=temperature,
                api_key=self._api_key,
                api_base=self._base_url,
                max_tokens=self._max_tokens,
                custom_llm_provider=self._custom_llm_provider,
            )
        return self._llm_cache[key]

    @staticmethod
    def _to_langchain_messages(messages: list[dict]) -> list:
        """Converts our OpenAI-shaped message dicts (the format used
        throughout run_agent's message history) into LangChain message
        objects. The one non-trivial part: our stored assistant tool_calls
        carry `arguments` as a JSON STRING (OpenAI's wire format), but
        LangChain's AIMessage.tool_calls expects `args` as an already-parsed
        dict -- silently getting this wrong wouldn't error, it would just
        send the model a malformed tool call history."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        converted = []
        for m in messages:
            role = m["role"]
            if role == "system":
                converted.append(SystemMessage(content=m["content"]))
            elif role == "user":
                converted.append(HumanMessage(content=m["content"]))
            elif role == "tool":
                converted.append(ToolMessage(content=m["content"], tool_call_id=m["tool_call_id"]))
            elif role == "assistant":
                tool_calls = []
                for tc in m.get("tool_calls") or []:
                    args = json.loads(tc["function"]["arguments"])
                    tool_calls.append(
                        {"name": tc["function"]["name"], "args": args, "id": tc["id"]}
                    )
                converted.append(AIMessage(content=m.get("content") or "", tool_calls=tool_calls))
            else:
                raise ValueError(f"Unknown message role for LangChain conversion: {role!r}")
        return converted

    @staticmethod
    def _extract_text_from_content(content) -> str:
        """Some reasoning-capable models (observed via this LiteLLM proxy)
        return `content` as a LIST of typed blocks (e.g.
        [{"type":"thinking",...}, {"type":"text","text":...}]) instead of a
        plain string, when they emit a reasoning trace alongside their
        actual response. Extracts and joins just the text-bearing blocks,
        deliberately skipping thinking/reasoning blocks (that's the model's
        scratch work, not its answer), so downstream code always sees a
        plain string regardless of which shape a given model uses."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(p for p in parts if p)
        return str(content) if content else ""

    @staticmethod
    def _try_salvage_submit_answer_from_text(text: str):
        """A real failure mode found in practice: a reasoning-capable model
        emitted its final answer as JSON-shaped TEXT (matching
        submit_answer's schema) instead of an actual structured tool call --
        `content` was a list of blocks (thinking + text), and the text
        block itself was a JSON string like {"answer":..., "has_answer":...,
        "citations":[...]}. Left unhandled, run_agent's plain-text fallback
        path treats ANY non-tool-call response as a successful
        has_answer=True answer UNCONDITIONALLY, bypassing citation
        verification entirely -- observed concretely on an unanswerable
        query where the model's embedded JSON correctly said
        has_answer=false, but the bug would have silently overridden that to
        true. Detecting and parsing this pattern routes it through the
        normal, fully-verified submit_answer path instead -- the same kind
        of salvage already done for Groq's pseudo-XML tool-call format
        (_salvage_tool_call), applied to a different malformed shape from a
        different provider. Returns None (not salvageable) if the text
        doesn't parse as a dict with at least answer/has_answer keys --
        genuine plain-text responses should NOT be forced through this path."""
        stripped = text.strip()
        stripped = stripped.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(parsed, dict) or "answer" not in parsed or "has_answer" not in parsed:
            return None

        parsed.setdefault("citations", [])
        return parsed

    @staticmethod
    def _to_openai_shaped_response(ai_message):
        """Converts a LangChain AIMessage back into the SimpleNamespace
        shape the rest of this file expects (response.choices[0].message
        .tool_calls[i].function.name/.arguments as a JSON STRING, matching
        what execute_tool_call's `json.loads(tc.function.arguments)` call
        expects) -- same synthetic-response pattern already used by
        _salvage_tool_call for the Groq path."""
        from types import SimpleNamespace

        text_content = _LangChainChatCompletionsAdapter._extract_text_from_content(
            ai_message.content
        )

        tool_calls = [
            SimpleNamespace(
                id=tc.get("id") or f"lc_{uuid.uuid4().hex[:8]}",
                function=SimpleNamespace(name=tc["name"], arguments=json.dumps(tc["args"])),
            )
            for tc in (ai_message.tool_calls or [])
        ]

        if not tool_calls:
            salvaged = _LangChainChatCompletionsAdapter._try_salvage_submit_answer_from_text(
                text_content
            )
            if salvaged is not None:
                log.warning(
                    "Model emitted a submit_answer-shaped JSON as plain text/content blocks "
                    "instead of a real tool call -- salvaging into a proper submit_answer call "
                    "rather than falling through to the unverified plain-text answer path"
                )
                tool_calls = [
                    SimpleNamespace(
                        id=f"lc_salvaged_{uuid.uuid4().hex[:8]}",
                        function=SimpleNamespace(
                            name="submit_answer", arguments=json.dumps(salvaged)
                        ),
                    )
                ]
                text_content = None  # the salvaged tool call is the real response now

        message = SimpleNamespace(content=text_content or None, tool_calls=tool_calls or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def create(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0.2,
    ):
        llm = self._get_llm(model, temperature)
        lc_messages = self._to_langchain_messages(messages)

        # Plain-text generation (no tools) is a real, separate use case in
        # this project -- eval/eval_answers.py's judge (via
        # _GenerateContentAdapter) calls create() with only model/messages,
        # no tools at all, since judging is a plain completion, not a tool
        # call. Calling bind_tools([]) with an empty list is untested/
        # ambiguous behavior to rely on, so skip binding entirely rather
        # than guess it degrades gracefully.
        target = llm.bind_tools(tools, tool_choice=tool_choice) if tools else llm
        ai_message = target.invoke(lc_messages)
        return self._to_openai_shaped_response(ai_message)


class _ChatNamespace:
    """Just enough nesting (client.chat.completions.create(...)) to match
    the Groq SDK's attribute path -- run_agent calls
    client.chat.completions.create(...), so this adapter needs the same
    shape, not just the same create() signature."""

    def __init__(self, adapter: _LangChainChatCompletionsAdapter):
        self.completions = adapter


class LangChainLiteLLMClient:
    """Top-level object returned by get_litellm_client() -- exposes
    .chat.completions.create(...), matching client.chat.completions.create(...)
    call sites throughout this file, backed by LangChain's ChatLiteLLM
    instead of Groq's SDK."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_tokens: int = 8000,
        custom_llm_provider: str = "openai",
    ):
        self.chat = _ChatNamespace(
            _LangChainChatCompletionsAdapter(base_url, api_key, max_tokens, custom_llm_provider)
        )


def get_litellm_client():
    """Points LangChain's ChatLiteLLM at a company/personal LiteLLM proxy.
    Reads LITELLM_BASE_URL and LITELLM_API_KEY from the environment/.env
    (LITELLM_MAX_TOKENS optional, defaults to 8000), matching the existing
    GROQ_API_KEY/GOOGLE_API_KEY pattern -- the key is never hardcoded or
    passed through chat, only loaded from a local secret.

    Uses ChatLiteLLM (the `litellm` package's LangChain integration), not
    ChatOpenAI -- two earlier attempts are worth knowing about since they
    ruled out simpler options first: reusing Groq's own SDK class pointed
    at a different base_url failed with a persistent 405 (Groq's SDK makes
    Groq-specific assumptions). LangChain's ChatOpenAI (strict OpenAI
    wire-format) got further but hit a server-side 'NoneType has no
    attribute startswith' error on every model tried. A colleague with this
    exact proxy already working confirmed they use ChatLiteLLM specifically
    -- this proxy isn't a fully strict OpenAI-compatible target, which is
    the gap ChatLiteLLM (built to speak each backend provider's actual
    dialect, not just generic OpenAI shape) is designed to bridge."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        import langchain_litellm  # noqa: F401
        import litellm  # noqa: F401
    except ImportError:
        log.error(
            "langchain-litellm/litellm not installed. Run: pip install langchain-litellm litellm"
        )
        sys.exit(1)

    base_url = os.environ.get("LITELLM_BASE_URL")
    api_key = os.environ.get("LITELLM_API_KEY")
    if not base_url or not api_key:
        log.error("LITELLM_BASE_URL and/or LITELLM_API_KEY not set -- add both to your .env file.")
        sys.exit(1)

    max_tokens = int(os.environ.get("LITELLM_MAX_TOKENS", "8000"))
    custom_llm_provider = os.environ.get("LITELLM_CUSTOM_PROVIDER", "openai")

    return LangChainLiteLLMClient(
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        custom_llm_provider=custom_llm_provider,
    )


# model defaults paired with each provider -- overridable via --model on any
# script that exposes this dispatcher, this is just a sensible starting point
PROVIDER_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "litellm": "openai.gpt-oss-120b-1:0",
}


def get_client_for_provider(provider: str):
    if provider == "litellm":
        return get_litellm_client()
    return get_groq_client()


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
        session.trajectory.append({"tool": name, "args": args, "error": True})
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
        session.trajectory.append({"tool": name, "args": args, "error": True})
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
                cross_encoder=ctx.cross_encoder,
                episode_index=ctx.episode_index,
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
            if "error" not in result:
                session.info_verified_episodes.add(args["episode_id"])

        elif name == "get_context_window":
            result = get_context_window_tool(
                chunk_id=args["chunk_id"],
                corpus_by_id=ctx.corpus_by_id,
                episode_index=ctx.episode_index,
                before=args.get("before", 1),
                after=args.get("after", 1),
            )
            if "chunks" in result:
                for chunk in result["chunks"]:
                    session.seen_chunk_ids.add(chunk["chunk_id"])
                    session.retrieved_chunks[chunk["chunk_id"]] = {
                        "chunk_id": chunk["chunk_id"],
                        "episode_id": result["episode_id"],
                        "start": chunk["start"],
                        "end": chunk["end"],
                    }

        else:
            result = {"error": f"Unknown tool: {name}"}

    except Exception as e:
        log.warning("Tool %s failed: %s", name, e)
        result = {"error": str(e)}

    elapsed = time.time() - start
    session.tool_call_latencies.append({"tool": name, "seconds": round(elapsed, 3)})
    session.trajectory.append({"tool": name, "args": args, "error": "error" in result})
    if verbose:
        log.info("Tool call: %s(%s) [%.2fs]", name, args, elapsed)

    return result


EPISODE_INFO_CITATION_CHUNK_ID = "episode_info"  # sentinel chunk_id value for citing
# get_episode_info-sourced claims, which have no real chunk to point at -- see
# verify_citations for how this is checked differently from a real chunk citation


def _floats_close(a, b, tol: float = 0.5) -> bool:
    """Tolerant comparison for citation timestamp verification. A real bug
    found in practice: comparing chunk start/end with exact `==` silently
    invalidated genuinely correct citations -- observed on live data where
    2 of 3 citations from the same search call verified fine and a 3rd,
    identically-sourced one didn't, strongly suggesting float precision
    drift (JSON round-tripping, or the model re-typing a timestamp instead
    of copying it exactly) rather than an actual fabricated citation. Since
    chunk_id is already the authoritative match (an exact dict key lookup
    in session.retrieved_chunks), this check is a secondary sanity check
    against field-swapping, not the primary trust anchor -- half a second
    of tolerance is generous enough to absorb precision noise while still
    catching a genuinely wrong timestamp."""
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return False


def verify_citations(citations: list[dict], session: SessionState) -> list[dict]:
    """Checks each citation against chunks actually retrieved this session.
    A citation is verified if its chunk_id was really shown to the model AND
    the episode_id/timestamp claimed matches what that chunk actually is --
    catching both fabricated chunk_ids and a chunk_id paired with mismatched
    metadata (a subtler, easy-to-miss form of the same problem). Timestamp
    matching uses a tolerance, not exact equality -- see _floats_close.

    A citation with chunk_id == EPISODE_INFO_CITATION_CHUNK_ID is a
    different, valid category: an answer sourced from get_episode_info
    (episode metadata -- air date, tags, description) rather than a
    transcript chunk. There's no real chunk to match for these, so they're
    verified against whether get_episode_info actually succeeded for that
    episode_id this session instead. Without this, EVERY episode_info-type
    question was structurally unable to succeed regardless of how correct
    the model's answer was -- confirmed on live data across 3/3 such
    scenarios in one eval run, where the model got genuinely correct
    information (verified against episodes.yaml) every time and still had
    the answer discarded by the zero-valid-citation override.

    Also backfills guest (and any other display fields) from the actual
    retrieved chunk rather than trusting the model to have included them --
    `guest` is optional in the submit_answer schema, so relying on the model
    to remember it produces citations that display as "?" even when
    perfectly valid."""
    verified = []
    for c in citations:
        chunk_id = c.get("chunk_id")

        if chunk_id == EPISODE_INFO_CITATION_CHUNK_ID:
            is_valid = c.get("episode_id") in session.info_verified_episodes
            verified.append({**c, "verified": is_valid})
            continue

        chunk = session.retrieved_chunks.get(chunk_id)
        is_valid = (
            chunk is not None
            and chunk["episode_id"] == c.get("episode_id")
            and _floats_close(chunk["start"], c.get("start"))
            and _floats_close(chunk["end"], c.get("end"))
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
        remaining = MAX_ITERATIONS - iteration

        # Passive prompt guidance alone didn't change behavior in practice --
        # a real case (Claude Sonnet on a broad synthesis question) ran the
        # identical 5-search, non-converging pattern with or without the
        # "stop once you have enough" instruction in SYSTEM_PROMPT. An
        # explicit, escalating reminder as the budget actually depletes is a
        # more direct forcing function than a passive instruction the model
        # can keep deprioritizing turn after turn. Computed fresh each
        # iteration and appended only to THIS call's messages, not persisted
        # into `messages` -- so it reflects the current true remaining count
        # rather than stacking up duplicate reminders in the history.
        call_messages = messages
        if remaining == 2:
            call_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Reminder: you have 2 tool calls left in this session. If you "
                        "don't already have enough to answer, do at most one more search, "
                        "then call submit_answer with the best answer you can give from "
                        "what you've found -- do not keep researching further angles."
                    ),
                }
            ]
        elif remaining == 1:
            call_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "This is your LAST allowed tool call. You must call submit_answer "
                        "now with the best answer you can give from what you've already "
                        "found. Do not call any other tool."
                    ),
                }
            ]

        session.tokens_in += estimate_messages_tokens(call_messages)
        llm_start = time.time()
        try:
            response = call_llm_with_retry(
                client,
                model=model,
                messages=call_messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            session.llm_call_latencies.append(round(time.time() - llm_start, 3))
            is_daily_limit = getattr(e, "is_daily_limit", False)
            log.error("LLM call failed after retries: %s", e)
            if is_daily_limit:
                answer = (
                    "The Groq API daily token quota was exhausted while running this "
                    "query -- this is an infrastructure limit, not a reflection of "
                    "whether the corpus actually contains an answer. Wait for the quota "
                    "to reset (see the error for Groq's suggested wait time) or upgrade "
                    "tier, then re-run this specific query."
                )
            else:
                answer = (
                    "The model repeatedly failed to produce a valid tool call for this "
                    "question. This is usually transient -- try rephrasing the question "
                    "or running it again."
                )
            return _finalize(
                has_answer=False,
                answer=answer,
                citations=[],
                session=session,
                grounded=False,
                infra_failure=True,
                infra_failure_reason="daily_limit"
                if is_daily_limit
                else "tool_use_failed_exhausted",
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

        # submit_answer is only honored when it's the SOLE tool call in this
        # response. A real failure mode found on live data: the model bundled
        # list_episodes + search_transcripts (which failed validation) +
        # submit_answer all in one batch, without ever seeing the
        # list_episodes results -- which actually contained the real
        # episode_id it needed. Returning immediately on a bundled
        # submit_answer meant the model never got a chance to act on its own
        # tool results within the same turn, defeating the "read the error
        # and retry" instruction entirely. Deferring it forces a genuine next
        # iteration where the model has real results to react to.
        submit_call = next((tc for tc in tool_calls if tc.function.name == "submit_answer"), None)
        if submit_call and len(tool_calls) > 1:
            log.warning(
                "Model bundled submit_answer with %d other unresolved tool call(s) -- "
                "deferring it so those results are seen before a final answer is accepted",
                len(tool_calls) - 1,
            )

        for tc in tool_calls:
            args = json.loads(tc.function.arguments)

            if tc.function.name == "submit_answer":
                if len(tool_calls) > 1:
                    # Deferred: respond to this tool_call_id (required by the
                    # API -- every tool_call must get a matching result) but
                    # don't treat it as final. The other calls in this same
                    # batch still execute normally below.
                    session.trajectory.append(
                        {"tool": "submit_answer", "args": args, "error": False, "deferred": True}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(
                                {
                                    "deferred": True,
                                    "reason": (
                                        "submit_answer was called alongside other tool calls "
                                        "that haven't returned results yet. Wait for those "
                                        "results, then call submit_answer again on its own "
                                        "once you've actually used them."
                                    ),
                                }
                            ),
                        }
                    )
                    continue

                session.trajectory.append({"tool": "submit_answer", "args": args, "error": False})
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
    infra_failure: bool = False,
    infra_failure_reason: str | None = None,
) -> dict:
    unverified = [c for c in citations if not c.get("verified", True)]
    if unverified:
        log.warning(
            "%d citation(s) failed verification -- model cited chunks it wasn't shown",
            len(unverified),
        )

    return {
        "infra_failure": infra_failure,  # True if this reflects an API/infra failure
        # (daily quota exhausted, retries exhausted) rather than a genuine model
        # reasoning outcome -- eval scripts should exclude these from task-success
        # scoring rather than counting them as the agent having failed to answer
        "infra_failure_reason": infra_failure_reason,  # "daily_limit" | "tool_use_failed_exhausted" | None --
        # daily_limit is guaranteed to recur on every subsequent call until the quota
        # resets (hours), so callers running a batch (eval_agent.py) should stop
        # rather than burn through remaining scenarios that will fail identically.
        # tool_use_failed_exhausted is query-specific and doesn't predict the next call.
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
        "trajectory": session.trajectory,
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
    parser.add_argument(
        "--provider",
        type=str,
        choices=["groq", "litellm"],
        default="groq",
        help="groq (default, free tier) or litellm (a company/personal LiteLLM proxy -- "
        "requires LITELLM_BASE_URL and LITELLM_API_KEY in .env). Not free -- litellm "
        "models are billed per-token; see PROVIDER_DEFAULT_MODELS for the cheapest "
        "tool-calling-capable default.",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Overrides the provider's default model."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        settings = yaml.safe_load(f)

    model = args.model or PROVIDER_DEFAULT_MODELS[args.provider]

    ctx = AgentContext(settings, args.chunks, args.episodes_config)
    client = get_client_for_provider(args.provider)

    result = run_agent(args.query, ctx, client, model, verbose=args.verbose)

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
