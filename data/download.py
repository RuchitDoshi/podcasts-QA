"""
Download episode audio listed in config/episodes.yaml using yt-dlp.

Each episode's audio is extracted, resampled to mono 16kHz (matching the
convention in CLAUDE.md), and saved to data/raw/{episode_id}.wav.

Usage:
    python data/download.py --config config/episodes.yaml
    python data/download.py --config config/episodes.yaml --episode ep001
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("download")


def load_episodes(config_path: Path) -> list[dict]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    episodes = cfg.get("episodes", [])
    if not episodes:
        log.warning("No episodes found in %s — fill in config/episodes.yaml first.", config_path)
    return episodes


def download_episode(episode: dict, raw_dir: Path, sample_rate: int = 16000) -> Path | None:
    try:
        import yt_dlp
    except ImportError:
        log.error("yt-dlp is not installed. Run: pip install yt-dlp")
        sys.exit(1)

    episode_id = episode["id"]
    url = episode["url"]
    out_path = raw_dir / f"{episode_id}.wav"

    if out_path.exists():
        log.info("Skipping %s — already downloaded at %s", episode_id, out_path)
        return out_path

    if "REPLACE_ME" in url:
        log.warning(
            "Episode %s still has a placeholder URL — edit config/episodes.yaml", episode_id
        )
        return None

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(raw_dir / f"{episode_id}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "postprocessor_args": [
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
        ],
        "quiet": False,
        "noprogress": False,
    }

    log.info("Downloading %s (%s)", episode_id, url)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        log.error("Failed to download %s: %s", episode_id, e)
        return None

    if not out_path.exists():
        log.error("Expected output not found for %s at %s", episode_id, out_path)
        return None

    log.info("Saved %s", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download episode audio via yt-dlp")
    parser.add_argument("--config", type=Path, default=Path("config/episodes.yaml"))
    parser.add_argument("--output", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--episode", type=str, default=None, help="Only download this episode id")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(args.config)

    if args.episode:
        episodes = [e for e in episodes if e["id"] == args.episode]
        if not episodes:
            log.error("Episode id %s not found in %s", args.episode, args.config)
            sys.exit(1)

    results = []
    for episode in episodes:
        path = download_episode(episode, args.output)
        results.append((episode["id"], path is not None))

    ok = sum(1 for _, success in results if success)
    log.info("Done: %d/%d episodes downloaded successfully", ok, len(results))
    failed = [eid for eid, success in results if not success]
    if failed:
        log.warning("Failed or skipped: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
