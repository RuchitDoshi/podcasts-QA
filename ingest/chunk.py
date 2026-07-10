"""
Merge Whisper segments into larger, retrieval-friendly chunks, anchored on
time windows and (when available) speaker turns -- not naive fixed-size text
splitting, which breaks badly on spoken content (disfluencies, short VAD-
driven segments, mid-sentence cuts).

Chunking strategy:
- Accumulate consecutive segments into a chunk until either:
    (a) target_seconds of speech is reached AND a speaker change occurs, or
    (b) max_seconds is hit regardless of speaker (hard cap), or
    (c) the segment's ad/not-ad status differs from the current chunk's --
        this is a hard boundary, not a preference, so a sponsor read never
        gets merged into (and diluted within) a real-content chunk, or
        vice versa, or
    (d) input segments run out
- If diarization hasn't been run (all speakers null), this falls back to pure
  time-windowed chunking -- still timestamp-anchored, just without the
  speaker-turn preference.
- A small overlap_seconds of trailing context is carried into the next chunk,
  so an answer whose evidence straddles a chunk boundary is still retrievable
  from either chunk.

Ad detection is done at the segment level (short text, so a single pattern
hit is a strong signal) rather than as a post-hoc density check over the
merged chunk -- an earlier density-based, chunk-level version of this let
short ad reads dilute below threshold when merged with adjacent real
content, causing real sponsor reads to be misflagged as clean.

Detection has three layers, in order of how much they can catch:
1. Generic keyword patterns (AD_PATTERNS below) -- catches boilerplate ad
   language regardless of which sponsor it is.
2. Sponsor name + promotional cue co-occurrence -- if ingest/sponsor_extractor.py
   has been run, config/sponsors.json maps each episode to the sponsor
   companies actually read out in its intro. A segment naming a known
   sponsor AND containing a promotional cue word is flagged. Company name
   ALONE is deliberately not sufficient -- a guest could legitimately
   discuss a sponsor's product as real conversation content, so requiring
   both avoids that false positive.
3. ingest/ad_classifier.py's per-window LLM classification (optional,
   --use-llm-ad-detection) -- catches whatever both of the above miss, at
   higher cost (hundreds of calls vs. sponsor extraction's ~30).
Layers 1 and 2 run by default and cost nothing beyond one one-time sponsor
extraction pass; layer 3 is opt-in and now mostly a safety net rather than
the primary strategy, since a known-sponsor-name match is both cheaper and
more precise than asking an LLM to judge each chunk from scratch.

Each chunk is enriched with episode-level metadata (guest, date, tags) from
config/episodes.yaml, so the agent's retrieval/filtering layer doesn't need a
second lookup at query time.

Usage:
    python ingest/chunk.py --transcripts data/transcripts --output data/chunks
    python ingest/chunk.py --transcripts data/transcripts --output data/chunks --episode ep001
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("chunk")

# Generic boilerplate patterns -- catch ad language regardless of sponsor
# identity. Applied per-segment (short text), so a single match is treated
# as a strong signal rather than needing a density threshold.
AD_PATTERNS = [
    r"\bsponsor(s|ed)?\b",
    r"\bbrought to you by\b",
    r"\bpromo code\b",
    r"\buse code\b",
    r"\bcheck (them|it) out\b",
    r"\bdiscount\b",
    r"% off\b",
    r"\bfree trial\b",
    r"\.com slash\b",
    r"\.com/[a-z]+\b",
    r"\bsign up\b",
    r"\bvisit [a-z0-9]+\.(com|ai|io)\b",
]
AD_PATTERN_RE = re.compile("|".join(AD_PATTERNS), re.IGNORECASE)

# Lighter-weight promotional cues -- individually too common/generic to
# trust alone (e.g. "code" or "offer" show up in normal conversation too),
# but safe to use once gated by requiring a known sponsor NAME in the same
# segment (see is_ad_segment). This is what catches phrasing the strict
# AD_PATTERNS above would miss, e.g. the real case that motivated this
# layer: a FINN sponsor read ("...maximizes the average resolution rate...
# trusted by over 5000 customer service leaders...") that named no obvious
# ad keyword from the original strict list at all.
PROMO_CUE_PATTERNS = [
    r"\bsponsor\b",
    r"\bcode\b",
    r"\bdiscount\b",
    r"% off\b",
    r"\bfree trial\b",
    r"\bcheck (it|them) out\b",
    r"\bsign up\b",
    r"\bvisit\b",
    r"\btrusted by\b",
    r"\bpromo\b",
    r"\boffer\b",
    r"\bslash\b",
    r"\.com\b",
    r"\bbrought to you by\b",
    r"\bresolution rate\b",
    r"\bcustomer service\b",
]
PROMO_CUE_RE = re.compile("|".join(PROMO_CUE_PATTERNS), re.IGNORECASE)


def build_sponsor_regex(sponsor_names: list[str]) -> re.Pattern | None:
    """Compiles a regex matching any of the given sponsor company names,
    case-insensitive with word boundaries. Returns None for an empty list
    so callers can skip the sponsor-check branch entirely rather than
    matching against a pattern that can never hit.

    Whitespace within a multi-word name (e.g. "Better Help") is matched as
    \\s+ rather than a literal space, so "BetterHelp" (no space, a plausible
    ASR or LLM-normalization variant) and "Better  Help" (double space)
    still match -- exact-substring matching on a literal space was found to
    silently fail on both of those during testing, and there are two
    independent places that kind of spacing drift can creep in: Whisper's
    transcription of a brand name, or the sponsor-extraction LLM
    normalizing to a brand's canonical spelling instead of preserving
    whatever the transcript literally said."""
    if not sponsor_names:
        return None
    escaped = [re.escape(name).replace(r"\ ", r"\s*") for name in sponsor_names if name.strip()]
    if not escaped:
        return None
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


def is_ad_segment(text: str, sponsor_re: re.Pattern | None = None) -> bool:
    if AD_PATTERN_RE.search(text):
        return True
    if sponsor_re and sponsor_re.search(text) and PROMO_CUE_RE.search(text):
        return True
    return False


def load_settings(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_episode_metadata(episodes_config: Path) -> dict[str, dict]:
    with open(episodes_config) as f:
        data = yaml.safe_load(f)
    return {ep["id"]: ep for ep in data.get("episodes", [])}


def tag_ad_segments(segments: list[dict], sponsor_re: re.Pattern | None) -> None:
    """Tags each segment's "_is_ad" in place, in two passes.

    Pass 1: check each segment's own text in isolation, as before.

    Pass 2: a real miss found on live data -- a sponsor read ("...this
    episode is brought to you by Tax Network USA...") landed across a
    Whisper segment boundary, e.g. one segment ending "...brought to you"
    and the next starting "by Tax Network USA...". Neither segment alone
    contains the complete trigger phrase, so pass 1 misses both, and
    because nothing gets flagged, the hard chunk-boundary logic never
    fires and the whole sponsor read merges into a large real-content
    chunk instead of being isolated. Pass 2 checks each adjacent pair's
    CONCATENATED text; if the pair matches but neither segment matched on
    its own, that's specifically a split-phrase case (not a coincidental
    match), so both segments get flagged. Checking "did the pair match but
    neither half alone" rather than just "does the pair match" avoids
    double-flagging pairs where segment i+1 already matched independently
    for an unrelated reason."""
    for seg in segments:
        seg["_is_ad"] = is_ad_segment(seg["text"], sponsor_re)

    for i in range(len(segments) - 1):
        seg_a, seg_b = segments[i], segments[i + 1]
        if seg_a["_is_ad"] or seg_b["_is_ad"]:
            continue  # already covered, or will be covered by another pair
        pair_text = seg_a["text"] + " " + seg_b["text"]
        if is_ad_segment(pair_text, sponsor_re):
            seg_a["_is_ad"] = True
            seg_b["_is_ad"] = True


def chunk_segments(
    segments: list[dict],
    settings: dict,
    sponsor_names: list[str] | None = None,
    llm_client=None,
    llm_model: str | None = None,
) -> list[dict]:
    c = settings["chunking"]
    target_seconds = c["target_seconds"]
    max_seconds = c["max_seconds"]
    overlap_seconds = c["overlap_seconds"]

    if not segments:
        return []

    # Tag ad status up front so the boundary check below is a cheap lookup
    # rather than re-running detection per comparison. Generic regex + the
    # sponsor-name/promo-cue combo both always run (fast, free once sponsors
    # are extracted); if an LLM client is also supplied, classify_segments_llm
    # runs as a further pass that can only ADD detections the above missed
    # (see ad_classifier.py's OR-safety property), never remove one.
    sponsor_re = build_sponsor_regex(sponsor_names or [])
    tag_ad_segments(segments, sponsor_re)

    if llm_client is not None:
        from ingest.ad_classifier import classify_segments_llm

        segments = classify_segments_llm(segments, llm_client, llm_model)

    chunks = []
    current = []
    current_start = segments[0]["start"]

    def flush(next_start_idx: int, keep_overlap: bool = True) -> int:
        """Close out the current chunk, seed the next one with overlap. Returns
        the index into `segments` to resume from (accounting for overlap)."""
        nonlocal current, current_start
        if not current:
            return next_start_idx

        chunk_end = current[-1]["end"]
        ad_segment_count = sum(1 for seg in current if seg["_is_ad"])
        chunks.append(
            {
                "start": current_start,
                "end": chunk_end,
                "text": " ".join(seg["text"] for seg in current).strip(),
                "speakers": sorted({s for seg in current if (s := seg.get("speaker"))}),
                "is_likely_ad": ad_segment_count / len(current) >= 0.5,
            }
        )

        if keep_overlap:
            # Find how many trailing segments fall within overlap_seconds of the
            # chunk end -- carry those into the next chunk for continuity. Only
            # segments matching the chunk's own last segment's ad status are
            # carried, so the carry itself is never internally mixed.
            overlap_from = chunk_end - overlap_seconds
            last_is_ad = current[-1]["_is_ad"]
            carry = [
                seg for seg in current if seg["end"] > overlap_from and seg["_is_ad"] == last_is_ad
            ]
        else:
            # Forced no-carry flush: used when the carry from the FIRST flush
            # attempt still conflicts with the segment that's about to be
            # appended next. That carry is internally homogeneous (all one ad
            # status, by construction above) but that status can still be the
            # opposite of the incoming segment -- exactly the situation that
            # triggered the flush in the first place. A real case this fixed:
            # a sponsor read split across a Whisper segment boundary ("...this
            # episode is brought to you" | "by Tax Network USA...") followed
            # immediately by real content -- the ad fragment kept getting
            # carried forward and silently merged with the next real-content
            # segment because the carry check only compared against the OLD
            # last segment, never against what was about to be appended.
            # Carrying nothing here breaks that cycle; there's no continuity
            # benefit anyway since the fragment and the incoming segment sit
            # on opposite sides of a hard ad/content boundary.
            carry = []

        current = list(carry)
        current_start = carry[0]["start"] if carry else chunk_end
        return next_start_idx

    for i, seg in enumerate(segments):
        prev_speaker = current[-1].get("speaker") if current else None
        seg_speaker = seg.get("speaker")
        speaker_changed = (
            prev_speaker is not None and seg_speaker is not None and seg_speaker != prev_speaker
        )

        prev_is_ad = current[-1]["_is_ad"] if current else None
        ad_status_changed = prev_is_ad is not None and seg["_is_ad"] != prev_is_ad

        would_be_duration = seg["end"] - current_start if current else 0

        if current and ad_status_changed:
            flush(i)
            if current and current[-1]["_is_ad"] != seg["_is_ad"]:
                # The overlap carry from the flush above is still incompatible
                # with the segment we're about to append -- discard the carry
                # entirely rather than let it silently re-merge the two sides
                # of the boundary we just tried to separate.
                flush(i, keep_overlap=False)
        elif current and would_be_duration >= max_seconds:
            flush(i)
        elif current and would_be_duration >= target_seconds and speaker_changed:
            flush(i)

        current.append(seg)

    if current:
        ad_segment_count = sum(1 for seg in current if seg["_is_ad"])
        chunks.append(
            {
                "start": current_start,
                "end": current[-1]["end"],
                "text": " ".join(seg["text"] for seg in current).strip(),
                "speakers": sorted({s for seg in current if (s := seg.get("speaker"))}),
                "is_likely_ad": ad_segment_count / len(current) >= 0.5,
            }
        )

    return chunks


def build_chunk_records(episode_id: str, chunks: list[dict], metadata: dict | None) -> list[dict]:
    meta = metadata or {}
    records = []
    for idx, chunk in enumerate(chunks):
        records.append(
            {
                "chunk_id": f"{episode_id}__chunk{idx:04d}",
                "episode_id": episode_id,
                "start": round(chunk["start"], 2),
                "end": round(chunk["end"], 2),
                "text": chunk["text"],
                "metadata": {
                    "title": meta.get("title"),
                    "guest": meta.get("guest"),
                    "date": meta.get("date"),
                    "tags": meta.get("tags", []),
                    "speakers": chunk["speakers"],
                    "is_likely_ad": chunk["is_likely_ad"],
                },
            }
        )
    return records


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
        log.error("GROQ_API_KEY not set -- required for --use-llm-ad-detection.")
        sys.exit(1)

    return Groq(api_key=api_key)


def main():
    parser = argparse.ArgumentParser(description="Chunk transcripts into retrieval-ready segments")
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-transcripts"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-chunks"),
    )
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--episodes-config", type=Path, default=Path("config/episodes.yaml"))
    parser.add_argument("--episode", type=str, default=None, help="Only chunk this episode id")
    parser.add_argument(
        "--use-llm-ad-detection",
        action="store_true",
        help=(
            "Also classify ad content via LLM (Groq), catching sponsor scripts the regex "
            "patterns don't recognize. Slow on free tier -- roughly an hour for the full "
            "corpus at safe rate-limit pacing (see ad_classifier.py). Regex-only detection "
            "still runs first and is never weakened by this."
        ),
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama-3.1-8b-instant",
        help="A small/fast model is plenty for this yes/no classification task and "
        "tends to have friendlier free-tier limits than the larger models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-chunk episodes even if their output file already exists. Without this, "
        "episodes with an existing chunk file are skipped -- important when "
        "--use-llm-ad-detection is slow enough that a run may not finish in one sitting.",
    )
    parser.add_argument(
        "--sponsors-file",
        type=Path,
        default=Path("config/sponsors.json"),
        help="Output of ingest/sponsor_extractor.py -- per-episode sponsor company names, "
        "used for the sponsor-name + promotional-cue detection layer. If missing, that "
        "layer is silently skipped and detection falls back to generic patterns only.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    settings = load_settings(args.config)
    episode_meta = load_episode_metadata(args.episodes_config)

    sponsors_by_episode: dict[str, list[str]] = {}
    if args.sponsors_file.exists():
        with open(args.sponsors_file) as f:
            sponsors_by_episode = json.load(f).get("per_episode", {})
        log.info(
            "Loaded sponsor names for %d episodes from %s",
            len(sponsors_by_episode),
            args.sponsors_file,
        )
    else:
        log.warning(
            "%s not found -- sponsor-name detection layer disabled. Run "
            "ingest/sponsor_extractor.py first to enable it.",
            args.sponsors_file,
        )

    llm_client = get_groq_client() if args.use_llm_ad_detection else None
    if llm_client:
        log.info("LLM-based ad detection enabled (model=%s)", args.llm_model)

    transcript_files = sorted(args.transcripts.glob("*.json"))
    if args.episode:
        transcript_files = [f for f in transcript_files if f.stem == args.episode]
        if not transcript_files:
            log.error("No transcript found for episode id %s", args.episode)
            sys.exit(1)

    if not transcript_files:
        log.warning("No transcripts found in %s -- run data/transcribe.py first.", args.transcripts)
        return

    total_chunks = 0
    skipped = 0
    for transcript_path in transcript_files:
        episode_id_guess = transcript_path.stem
        out_path = args.output / f"{episode_id_guess}.json"
        if out_path.exists() and not args.overwrite:
            log.info(
                "Skipping %s -- chunk file already exists (use --overwrite to redo)",
                episode_id_guess,
            )
            skipped += 1
            continue

        with open(transcript_path) as f:
            transcript = json.load(f)

        episode_id = transcript["episode_id"]
        chunks = chunk_segments(
            transcript["segments"],
            settings,
            sponsor_names=sponsors_by_episode.get(episode_id, []),
            llm_client=llm_client,
            llm_model=args.llm_model,
        )
        records = build_chunk_records(episode_id, chunks, episode_meta.get(episode_id))

        out_path = args.output / f"{episode_id}.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)

        avg_len = sum(r["end"] - r["start"] for r in records) / len(records) if records else 0
        ad_count = sum(1 for r in records if r["metadata"]["is_likely_ad"])
        log.info(
            "%s: %d segments -> %d chunks (avg %.0fs, %d flagged as ads) -> %s",
            episode_id,
            len(transcript["segments"]),
            len(records),
            avg_len,
            ad_count,
            out_path,
        )
        total_chunks += len(records)

    log.info(
        "Done: %d episodes chunked, %d skipped (already existed), %d total chunks",
        len(transcript_files) - skipped,
        skipped,
        total_chunks,
    )


if __name__ == "__main__":
    main()
