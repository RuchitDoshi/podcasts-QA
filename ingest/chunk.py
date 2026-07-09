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

# Sponsor reads follow a fairly formulaic pattern across episodes ("brought to
# you by", "promo code", "check them out", a bare URL/domain mention, etc.).
# Applied per-segment (short text), so a single match is treated as a strong
# signal rather than needing a density threshold.
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


def is_ad_segment(text: str) -> bool:
    return bool(AD_PATTERN_RE.search(text))


def load_settings(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_episode_metadata(episodes_config: Path) -> dict[str, dict]:
    with open(episodes_config) as f:
        data = yaml.safe_load(f)
    return {ep["id"]: ep for ep in data.get("episodes", [])}


def chunk_segments(segments: list[dict], settings: dict) -> list[dict]:
    c = settings["chunking"]
    target_seconds = c["target_seconds"]
    max_seconds = c["max_seconds"]
    overlap_seconds = c["overlap_seconds"]

    if not segments:
        return []

    # Tag ad status once up front so the boundary check below is a cheap
    # lookup rather than re-running the regex per comparison.
    for seg in segments:
        seg["_is_ad"] = is_ad_segment(seg["text"])

    chunks = []
    current = []
    current_start = segments[0]["start"]

    def flush(next_start_idx: int) -> int:
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

        # Find how many trailing segments fall within overlap_seconds of the
        # chunk end -- carry those into the next chunk for continuity. Skip
        # the overlap carry entirely across an ad/content boundary so we
        # don't immediately re-mix the two chunks we just tried to separate.
        overlap_from = chunk_end - overlap_seconds
        last_is_ad = current[-1]["_is_ad"]
        carry = [
            seg for seg in current if seg["end"] > overlap_from and seg["_is_ad"] == last_is_ad
        ]

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
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    settings = load_settings(args.config)
    episode_meta = load_episode_metadata(args.episodes_config)

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
    for transcript_path in transcript_files:
        with open(transcript_path) as f:
            transcript = json.load(f)

        episode_id = transcript["episode_id"]
        chunks = chunk_segments(transcript["segments"], settings)
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

    log.info("Done: %d episodes chunked, %d total chunks", len(transcript_files), total_chunks)


if __name__ == "__main__":
    main()
