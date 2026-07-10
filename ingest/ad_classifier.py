"""
LLM-based sponsor/ad detection, replacing the regex heuristic in chunk.py
for cases the keyword list can't generalize to.

Why this exists: the regex approach (AD_PATTERNS in chunk.py) only catches
phrasings it was explicitly written against. A real example that slipped
through: "...integrating AI in a way that maximizes the average resolution
rate" -- a FINN sponsor read reusing a script with different wording than
whatever the regex was tuned on. Keyword matching is inherently a losing
game against endlessly-rephrased ad copy; an LLM classifier generalizes to
scripts it's never seen, at the cost of an API call instead of a regex match.

Cost/granularity tradeoff: classifying every individual Whisper segment
would be prohibitively expensive -- some episodes have 20,000+ fragmented
segments (see the Tao/DHH/Plummer/ffmpeg VAD-fragmentation finding earlier
in this project). Instead, segments are grouped into ~20s probe windows
first, and multiple windows are classified in a single batched LLM call
(default 15 windows/call), which keeps the total call count low (roughly
a few hundred calls for the full 30-episode corpus, not tens of thousands).

Output shape: a per-segment is_ad boolean, matching what chunk.py's
chunk_segments() already expects from the regex path -- every window's
label is broadcast to all segments inside that window. This means
chunk_segments()'s hard-boundary logic doesn't need to change at all; only
the source of the boolean does.

Usage (as a library, called from chunk.py via --use-llm-ad-detection):
    from ingest.ad_classifier import classify_segments_llm
    segments = classify_segments_llm(segments, client, model, window_seconds=20)
"""

import json
import logging
import time

log = logging.getLogger("ad_classifier")

CLASSIFY_SYSTEM_PROMPT = """You classify short windows of a podcast transcript as \
either sponsor/advertisement content or real conversational content.

A window is an AD if it contains: sponsor reads, promo codes, discount offers, \
phrases like "brought to you by", calls to visit a URL or sign up for a product, \
or any paid promotional copy -- even if it doesn't use an obvious keyword like \
"sponsor" or "discount". Ad copy often sounds like marketing language: mentions \
of specific companies/products in a promotional tone, statistics about a \
product's performance, calls to action.

A window is NOT an ad if it's the host or guest discussing the episode's actual \
topic, even if they happen to mention a company or product name in a normal \
conversational way (e.g. a guest who works at a company discussing their own \
work is not an ad).

Respond with only a JSON array of booleans, one per numbered window, in order. \
No other text."""


def group_into_windows(segments: list[dict], window_seconds: float) -> list[dict]:
    """Groups consecutive segments into fixed-duration windows purely for ad
    classification -- these are NOT the final retrieval chunks, just a
    coarser unit that keeps LLM call volume manageable regardless of how
    fragmented the underlying Whisper segments are."""
    if not segments:
        return []

    windows = []
    current = []
    window_start = segments[0]["start"]

    for seg in segments:
        if current and seg["end"] - window_start > window_seconds:
            windows.append(
                {
                    "start": window_start,
                    "end": current[-1]["end"],
                    "segments": current,
                    "text": " ".join(s["text"] for s in current).strip(),
                }
            )
            current = []
            window_start = seg["start"]
        current.append(seg)

    if current:
        windows.append(
            {
                "start": window_start,
                "end": current[-1]["end"],
                "segments": current,
                "text": " ".join(s["text"] for s in current).strip(),
            }
        )

    return windows


def _is_rate_limit_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status == 429:
        return True
    body = getattr(error, "body", None)
    if isinstance(body, dict) and (body.get("error") or {}).get("code") == "rate_limit_exceeded":
        return True
    msg = str(error).lower()
    return "rate limit" in msg or "429" in msg


def _classify_batch(windows: list[dict], client, model: str, max_retries: int = 4) -> list[bool]:
    """Classifies one batch of windows in a single LLM call.

    Rate limit errors (429) are retried with backoff -- these are expected
    and recoverable on Groq's free tier given this pipeline's call volume,
    and treating them as a permanent failure (as an earlier version of this
    did) silently lost LLM classification coverage on whatever windows
    happened to be rate-limited, without any visible sign it had happened
    beyond a log line easy to miss across hundreds of calls.

    Any other failure (malformed response, non-JSON, genuine API error)
    still falls back to all-False (defer to regex) rather than retrying --
    those aren't likely to resolve on retry and shouldn't stall the pipeline."""
    numbered = "\n\n".join(f"[{i}] {w['text']}" for i, w in enumerate(windows))
    user_prompt = f"Classify these {len(windows)} windows:\n\n{numbered}"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=50 + len(windows) * 6,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            labels = json.loads(raw)

            if not isinstance(labels, list) or len(labels) != len(windows):
                log.warning(
                    "Ad classifier returned %d labels for %d windows -- discarding batch, "
                    "falling back to regex labels for these windows",
                    len(labels) if isinstance(labels, list) else -1,
                    len(windows),
                )
                return [False] * len(windows)

            return [bool(x) for x in labels]

        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries - 1:
                wait = 5.0 * (attempt + 1)  # 5s, 10s, 15s -- rate limit windows reset on the
                # order of seconds to a minute on Groq's free tier, not milliseconds
                log.warning(
                    "Rate limited on batch of %d windows, attempt %d/%d -- waiting %.0fs",
                    len(windows),
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue

            log.warning("Ad classification batch failed (%s) -- falling back to regex labels", e)
            return [False] * len(windows)

    log.warning("Ad classification batch exhausted retries -- falling back to regex labels")
    return [False] * len(windows)


def classify_segments_llm(
    segments: list[dict],
    client,
    model: str,
    window_seconds: float = 20.0,
    batch_size: int = 10,
    sleep_between_batches: float = 6.0,
) -> list[dict]:
    """Returns a new segments list with each segment's "_is_ad" set from its
    containing window's LLM classification, OR'd with the regex result if
    the segment already had "_is_ad" set (e.g. by chunk.py's regex pass) --
    this means the LLM classifier can only ADD detections the regex missed,
    never remove a regex-confirmed match, which keeps the combination
    strictly safer than either method alone.

    Pacing note: Groq's free tier caps at roughly 6000 tokens/minute (TPM),
    which is the actual binding constraint here, not the 30 requests/minute
    (RPM) cap -- a batch_size=10 call uses ~500 tokens, so sustained
    throughput is really ~11-12 calls/min regardless of how fast RPM would
    allow. The default sleep_between_batches=6.0 targets that budget with
    margin. Running this across the full 30-episode corpus (roughly a
    thousand windows total after grouping) takes on the order of an hour at
    these defaults -- expect it to be slow, that's the free-tier tradeoff,
    not a bug. Rate-limit (429) responses are retried with backoff rather
    than treated as permanent failures (see _classify_batch), so a
    temporarily-too-fast run degrades to slower, not to silently incomplete.

    Efficiency: a window is skipped entirely (never sent to the LLM) if
    every segment inside it is ALREADY regex-flagged as an ad. Since the OR
    combination means the LLM can only add detections, never remove one,
    classifying a fully-regex-covered window can't change its outcome --
    it would just spend an API call to confirm what's already known. Windows
    with at least one non-flagged segment are still sent, since the LLM
    might catch additional ad segments among those. In practice ad reads
    cluster at a few points per episode (open, occasional mid-roll) and a
    decent fraction of that text does contain an obvious regex trigger
    somewhere in it, so this typically cuts real request volume noticeably
    without losing any coverage."""
    windows = group_into_windows(segments, window_seconds)
    if not windows:
        return segments

    to_classify = [w for w in windows if not all(s["_is_ad"] for s in w["segments"])]
    skipped = len(windows) - len(to_classify)
    if skipped:
        log.info(
            "Skipping %d/%d windows already fully covered by regex -- sending %d to the LLM",
            skipped,
            len(windows),
            len(to_classify),
        )

    window_labels = []
    for i in range(0, len(to_classify), batch_size):
        batch = to_classify[i : i + batch_size]
        window_labels.extend(_classify_batch(batch, client, model))
        if i + batch_size < len(to_classify):
            time.sleep(sleep_between_batches)  # stay comfortably under free-tier rate limits

    for window, is_ad in zip(to_classify, window_labels, strict=True):
        if not is_ad:
            continue
        for seg in window["segments"]:
            seg["_is_ad"] = True  # OR with any existing regex-based label

    return segments
