"""
Extracts sponsor/advertiser company names per episode by asking an LLM to
read just the first few minutes of each transcript -- that's where Lex
reads the sponsor list on every episode, so there's no need to visit any
external site; the data we already have is the more accurate source (it
reflects what was actually read out in that specific episode, not just
whatever the current sponsor roster on the show's website happens to be).

This replaces per-chunk LLM ad classification (ad_classifier.py) as the
primary strategy: rather than asking an LLM "is this chunk an ad?" for
every chunk in the corpus (hundreds of calls, vague/vibes-based judgment),
this asks "who are the sponsors?" ONCE per episode (30 calls total, tiny
input each -- just the intro), then lets a cheap, precise regex do the
actual per-segment classification downstream (chunk.py) by checking for a
known sponsor name co-occurring with a promotional cue phrase. Company name
alone is not sufficient -- a guest could legitimately discuss a sponsor's
product as real content, so requiring BOTH a name and a promotional cue in
the same segment avoids that false-positive risk.

Usage:
    python ingest/sponsor_extractor.py --transcripts data/transcripts --output config/sponsors.json
    python ingest/sponsor_extractor.py --transcripts data/transcripts --output config/sponsors.json --episode ep001
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("sponsor_extractor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EXTRACT_SYSTEM_PROMPT = """You extract sponsor/advertiser company names from the \
opening minutes of a podcast transcript, where the host reads a list of paid \
sponsors before the real conversation begins.

Return only companies being read as PAID SPONSORS/ADVERTISERS -- not companies \
mentioned as topics of conversation, not the guest's own company, not competitors \
being discussed. If there are no sponsor reads in this text, return an empty array.

Respond with only a JSON array of company name strings, nothing else. Use the \
company's common short name (e.g. "Upwork" not "Upwork.com", "FINN" not "FINN AI").
Example: ["Upwork", "FINN", "BetterHelp"]"""


def get_intro_text(transcript: dict, max_seconds: float = 300.0) -> str:
    """Returns the concatenated text of segments within the first
    max_seconds of the episode -- sponsor reads are reliably an intro
    phenomenon in this corpus, so there's no need to scan the whole
    transcript (which would cost far more tokens for no benefit)."""
    segments = [s for s in transcript["segments"] if s["start"] < max_seconds]
    return " ".join(s["text"] for s in segments)


def _is_rate_limit_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status == 429:
        return True
    body = getattr(error, "body", None)
    if isinstance(body, dict) and (body.get("error") or {}).get("code") == "rate_limit_exceeded":
        return True
    msg = str(error).lower()
    return "rate limit" in msg or "429" in msg


def extract_sponsors_from_text(text: str, client, model: str, max_retries: int = 4) -> list[str]:
    if not text.strip():
        return []

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            names = json.loads(raw)

            if not isinstance(names, list):
                log.warning("Sponsor extraction returned non-list output: %r", raw)
                return []

            return sorted({str(n).strip() for n in names if str(n).strip()})

        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries - 1:
                wait = 5.0 * (attempt + 1)
                log.warning(
                    "Rate limited on sponsor extraction, attempt %d/%d -- waiting %.0fs",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue

            log.warning(
                "Sponsor extraction failed (%s) -- returning empty list for this episode", e
            )
            return []

    log.warning("Sponsor extraction exhausted retries -- returning empty list for this episode")
    return []


def extract_all_sponsors(
    transcripts_dir: Path,
    client,
    model: str,
    episode_filter: str | None = None,
    sleep_between_calls: float = 13.0,
) -> dict[str, list[str]]:
    """Pacing note: each call sends ~5 minutes of transcript text (~1200-1300
    tokens including the system prompt), so at Groq free tier's ~6000 TPM,
    sustained throughput is really only ~4-5 calls/min regardless of the 30
    RPM cap -- TPM is the binding constraint, same lesson as
    ad_classifier.py. sleep_between_calls=13.0 targets that budget with
    margin. For the full 30-episode corpus this takes on the order of 6-7
    minutes of sleep time alone, plus call latency -- much faster than the
    per-window LLM classifier since this is 30 calls total, not hundreds."""
    transcript_files = sorted(transcripts_dir.glob("*.json"))
    if episode_filter:
        transcript_files = [f for f in transcript_files if f.stem == episode_filter]

    per_episode: dict[str, list[str]] = {}
    for i, path in enumerate(transcript_files):
        with open(path) as f:
            transcript = json.load(f)

        episode_id = transcript["episode_id"]
        intro_text = get_intro_text(transcript)
        sponsors = extract_sponsors_from_text(intro_text, client, model)
        per_episode[episode_id] = sponsors
        log.info("%s: %s", episode_id, sponsors if sponsors else "(none found)")

        if i < len(transcript_files) - 1:
            time.sleep(sleep_between_calls)

    return per_episode


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

    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.error("GROQ_API_KEY not set.")
        sys.exit(1)

    return Groq(api_key=api_key)


def main():
    parser = argparse.ArgumentParser(description="Extract sponsor company names per episode")
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-transcripts"),
    )
    parser.add_argument("--output", type=Path, default=Path("config/sponsors.json"))
    parser.add_argument("--episode", type=str, default=None)
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant")
    parser.add_argument("--intro-seconds", type=float, default=600.0)
    parser.add_argument(
        "--sleep-between-calls",
        type=float,
        default=15.0,
        help="Seconds to wait between episode extraction calls, to stay under Groq free "
        "tier's ~6000 TPM limit (the actual binding constraint, not the 30 RPM cap).",
    )
    args = parser.parse_args()

    client = get_groq_client()
    per_episode = extract_all_sponsors(
        args.transcripts, client, args.model, args.episode, args.sleep_between_calls
    )

    # Merge with any existing file rather than clobbering results from a
    # prior partial run or a different --episode-scoped invocation.
    existing = {}
    if args.output.exists():
        with open(args.output) as f:
            existing = json.load(f)
    existing.update(per_episode)

    all_names = sorted({name for names in existing.values() for name in names})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"per_episode": existing, "all_sponsors": all_names}, f, indent=2)

    log.info(
        "Done: %d episodes, %d unique sponsor names found -> %s",
        len(per_episode),
        len(all_names),
        args.output,
    )
    log.info("All sponsors found: %s", all_names)


if __name__ == "__main__":
    main()
