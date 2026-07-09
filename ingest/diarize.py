"""
Run speaker diarization on episode audio and merge speaker labels into the
existing Whisper transcripts (data/transcripts/{episode_id}.json), filling in
the "speaker" field that transcribe.py leaves null.

Uses pyannote.audio's pretrained pipeline. This is a separate pass from
transcription on purpose -- diarization and ASR fail independently, and
keeping them decoupled means you can re-run one without redoing the other.

Requires:
- HF_TOKEN with access accepted for pyannote/speaker-diarization-3.1 (and its
  dependency pyannote/segmentation-3.0) at https://huggingface.co/pyannote
- pip install pyannote.audio

Alignment approach: diarization produces speaker turns as (start, end, speaker)
intervals independent of Whisper's segment boundaries. Each Whisper segment is
assigned the speaker whose turn overlaps it the most. This is a simple,
robust heuristic -- it can misassign the odd segment right at a speaker
changeover, but that's rare enough not to matter for retrieval quality.

Usage:
    python ingest/diarize.py --transcripts data/transcripts --audio data/raw
    python ingest/diarize.py --transcripts data/transcripts --audio data/raw --episode ep001
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("diarize")


def load_settings(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_pipeline(settings: dict):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log.error("pyannote.audio not installed. Run: pip install pyannote.audio")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log.error(
            "HF_TOKEN not set. Diarization models are gated -- accept access at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 first, then "
            "set HF_TOKEN in your environment or .env file."
        )
        sys.exit(1)

    d = settings["diarization"]
    log.info("Loading diarization pipeline: %s (device=%s)", d["model"], d["device"])

    pipeline = Pipeline.from_pretrained(d["model"], token=hf_token)

    if d["device"] == "cuda":
        import torch

        pipeline.to(torch.device("cuda"))

    return pipeline


def diarize_audio(pipeline, audio_path: Path, settings: dict) -> list[dict]:
    d = settings["diarization"]
    kwargs = {}
    if d.get("min_speakers"):
        kwargs["min_speakers"] = d["min_speakers"]
    if d.get("max_speakers"):
        kwargs["max_speakers"] = d["max_speakers"]

    start_time = time.time()
    output = pipeline(str(audio_path), **kwargs)
    elapsed = time.time() - start_time

    # pyannote.audio 4.0+ (community-1 and newer) returns a result object with
    # a `.speaker_diarization` attribute yielding (turn, speaker) pairs.
    # pyannote.audio 3.x returned an Annotation directly, with itertracks()
    # yielding (turn, track_id, speaker) triples. Support both so this script
    # doesn't break again on the next pyannote upgrade.
    if hasattr(output, "speaker_diarization"):
        turns = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, speaker in output.speaker_diarization
        ]
    else:
        turns = [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in output.itertracks(yield_label=True)
        ]

    speakers = sorted({t["speaker"] for t in turns})
    log.info(
        "Diarized %s in %.1fs -- %d turns, %d speakers (%s)",
        audio_path.name,
        elapsed,
        len(turns),
        len(speakers),
        ", ".join(speakers),
    )
    return turns


def assign_speaker(segment: dict, turns: list[dict]) -> str | None:
    """Assign the speaker whose turn has the most overlap with this segment."""
    seg_start, seg_end = segment["start"], segment["end"]
    best_speaker, best_overlap = None, 0.0

    for turn in turns:
        overlap = min(seg_end, turn["end"]) - max(seg_start, turn["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    return best_speaker


def find_audio_file(audio_dir: Path, episode_id: str) -> Path | None:
    for ext in (".wav", ".mp3", ".m4a"):
        candidate = audio_dir / f"{episode_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Diarize episode audio and merge speaker labels into transcripts"
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-transcripts"),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-wavs"),
    )
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--episode", type=str, default=None, help="Only diarize this episode id")
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-diarize even if speaker labels already exist"
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    transcript_files = sorted(args.transcripts.glob("*.json"))

    if args.episode:
        transcript_files = [f for f in transcript_files if f.stem == args.episode]
        if not transcript_files:
            log.error("No transcript found for episode id %s in %s", args.episode, args.transcripts)
            sys.exit(1)

    if not transcript_files:
        log.warning("No transcripts found in %s -- run data/transcribe.py first.", args.transcripts)
        return

    pipeline = get_pipeline(settings)

    ok, skipped, failed = 0, 0, 0
    for transcript_path in transcript_files:
        log.info("Processing %s", transcript_path.name)
        with open(transcript_path) as f:
            transcript = json.load(f)

        already_diarized = any(seg.get("speaker") for seg in transcript["segments"])
        if already_diarized and not args.overwrite:
            log.info("Skipping %s -- already has speaker labels", transcript_path.stem)
            skipped += 1
            continue

        audio_path = find_audio_file(args.audio, transcript["episode_id"])
        if not audio_path:
            log.error(
                "No audio file found for %s in %s -- skipping", transcript["episode_id"], args.audio
            )
            failed += 1
            continue

        try:
            turns = diarize_audio(pipeline, audio_path, settings)
        except Exception as e:
            log.error("Diarization failed for %s: %s", transcript["episode_id"], e)
            failed += 1
            continue

        for segment in transcript["segments"]:
            segment["speaker"] = assign_speaker(segment, turns)

        with open(transcript_path, "w") as f:
            json.dump(transcript, f, indent=2)

        log.info("Updated %s with speaker labels", transcript_path.name)
        ok += 1

    log.info(
        "Done: %d diarized, %d skipped, %d failed (of %d total)",
        ok,
        skipped,
        failed,
        len(transcript_files),
    )


if __name__ == "__main__":
    main()
