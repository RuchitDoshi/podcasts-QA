"""
Transcribe episode audio using self-hosted faster-whisper, producing
word/segment-level timestamped JSON transcripts.

Reads model + device settings from config/settings.yaml so the same script
works unchanged on a local CPU or a Colab T4 (just flip whisper.device and
whisper.compute_type in the config).

Output schema (data/transcripts/{episode_id}.json):
{
  "episode_id": "ep001",
  "language": "en",
  "duration": 3421.4,
  "segments": [
    {"start": 0.0, "end": 4.2, "speaker": null, "text": "..."},
    ...
  ]
}

Speaker diarization is left null here — wire up a diarization pass
(e.g. pyannote.audio) in a follow-up script and merge speaker labels into
these segments before chunking. Keeping transcription and diarization
separate makes it easy to re-run one without the other.

Usage:
    python data/transcribe.py --input data/raw --output data/transcripts
    python data/transcribe.py --input data/raw --output data/transcripts --episode ep001
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("transcribe")


def load_settings(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model(settings: dict):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log.error("faster-whisper is not installed. Run: pip install faster-whisper")
        sys.exit(1)

    w = settings["whisper"]
    log.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s",
        w["model_size"],
        w["device"],
        w["compute_type"],
    )
    return WhisperModel(w["model_size"], device=w["device"], compute_type=w["compute_type"])


def transcribe_file(model, audio_path: Path, settings: dict) -> dict:
    w = settings["whisper"]
    start_time = time.time()

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=w.get("language"),
        vad_filter=w.get("vad_filter", True),
        word_timestamps=True,
    )

    segments = []
    for seg in segments_iter:
        segments.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "speaker": None,  # filled in by a separate diarization pass
                "text": seg.text.strip(),
            }
        )

    elapsed = time.time() - start_time
    log.info(
        "Transcribed %s in %.1fs (%d segments, detected language=%s)",
        audio_path.name,
        elapsed,
        len(segments),
        info.language,
    )

    return {
        "episode_id": audio_path.stem,
        "language": info.language,
        "duration": round(info.duration, 2),
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with self-hosted Whisper")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/transcripts"))
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--episode", type=str, default=None, help="Only transcribe this episode id")
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-transcribe even if output exists"
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    settings = load_settings(args.config)

    audio_files = sorted(args.input.glob("*.wav"))
    if args.episode:
        audio_files = [f for f in audio_files if f.stem == args.episode]
        if not audio_files:
            log.error("No audio file found for episode id %s in %s", args.episode, args.input)
            sys.exit(1)

    if not audio_files:
        log.warning("No .wav files found in %s — run data/download.py first.", args.input)
        return

    model = get_model(settings)

    ok, skipped, failed = 0, 0, 0
    for audio_path in audio_files:
        log.info(f"Processing {audio_path.stem}...")
        out_path = args.output / f"{audio_path.stem}.json"
        if out_path.exists() and not args.overwrite:
            log.info("Skipping %s — transcript already exists at %s", audio_path.stem, out_path)
            skipped += 1
            continue

        try:
            transcript = transcribe_file(model, audio_path, settings)
        except Exception as e:
            log.error("Failed to transcribe %s: %s", audio_path.name, e)
            failed += 1
            continue

        with open(out_path, "w") as f:
            json.dump(transcript, f, indent=2)
        log.info("Wrote %s", out_path)
        ok += 1

    log.info(
        "Done: %d transcribed, %d skipped, %d failed (of %d total)",
        ok,
        skipped,
        failed,
        len(audio_files),
    )


if __name__ == "__main__":
    main()
