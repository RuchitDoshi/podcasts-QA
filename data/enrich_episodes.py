"""
Fix guest names and generate topic tags for episodes in config/episodes.yaml,
using the episode title + RSS description as context (not title alone).

Replaces the earlier tag_episodes.py: guest names parsed from the title with
a regex are wrong often enough to matter (multi-guest titles, debate-style
titles, question titles) -- the RSS description almost always states the
guest's identity in its first sentence, so it's a much better signal, and
it's already sitting in episodes.yaml if you ran the updated
pull_rss_episodes.py.

Requires GROQ_API_KEY in your environment (or a .env file, loaded via
python-dotenv).

Usage:
    python data/enrich_episodes.py --config config/episodes.yaml
    python data/enrich_episodes.py --config config/episodes.yaml --overwrite
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
log = logging.getLogger("enrich_episodes")

TAG_VOCAB_HINT = (
    "ai, machine_learning, robotics, physics, math, neuroscience, philosophy, "
    "history, politics, economics, war, space, biology, music, entrepreneurship, "
    "software_engineering, security, health, psychology, religion, sports, art"
)

SYSTEM_PROMPT = (
    "You extract structured metadata for podcast episodes from their title and "
    "description. Return strict JSON with exactly two keys:\n"
    '  "guest": a string with the guest name(s), comma-separated if there are '
    "multiple guests, or \"unknown\" if the description does not name one\n"
    '  "tags": an array of 2-5 lowercase snake_case topic tags\n'
    "Prefer tags from this vocabulary when they fit, but add a new one if none "
    "apply: " + TAG_VOCAB_HINT + "\n"
    "Return only the JSON object, nothing else."
)


def load_episodes(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_client():
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
        log.error("GROQ_API_KEY not set. Add it to your environment or a .env file.")
        sys.exit(1)

    return Groq(api_key=api_key)


def enrich_episode(client, episode: dict, model: str) -> dict | None:
    user_prompt = (
        f"Title: {episode.get('title', '')}\n"
        f"Description: {episode.get('description', '') or '(none available)'}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=150,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Could not parse enrichment output: %r", raw)
        return None

    if not isinstance(result, dict) or "guest" not in result or "tags" not in result:
        log.warning("Unexpected enrichment shape: %r", result)
        return None

    tags = result.get("tags") or []
    if not isinstance(tags, list):
        tags = []

    return {
        "guest": str(result.get("guest", "unknown")).strip(),
        "tags": [str(t).strip().lower() for t in tags if str(t).strip()],
    }


def main():
    parser = argparse.ArgumentParser(description="Fix guest names and generate tags using title + description")
    parser.add_argument("--config", type=Path, default=Path("config/episodes.yaml"))
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--overwrite", action="store_true",
                         help="Re-enrich episodes even if they already have a guest and tags")
    parser.add_argument("--sleep", type=float, default=1.0,
                         help="Seconds to sleep between calls, stay under free-tier rate limits")
    args = parser.parse_args()

    data = load_episodes(args.config)
    episodes = data.get("episodes", [])
    if not episodes:
        log.warning("No episodes found in %s", args.config)
        return

    client = get_client()

    updated = 0
    for episode in episodes:
        already_done = episode.get("tags") and episode.get("guest") not in (None, "", "unknown")
        if already_done and not args.overwrite:
            log.info("Skipping %s — already enriched (guest=%s, tags=%s)",
                      episode["id"], episode["guest"], episode["tags"])
            continue

        log.info("Enriching %s: %s", episode["id"], episode.get("title", ""))
        try:
            result = enrich_episode(client, episode, args.model)
        except Exception as e:
            log.error("Failed to enrich %s: %s", episode["id"], e)
            continue

        if result:
            episode["guest"] = result["guest"]
            episode["tags"] = result["tags"]
            updated += 1
            log.info("  -> guest=%s, tags=%s", result["guest"], result["tags"])
        else:
            log.warning("  -> enrichment failed for %s, leaving as-is", episode["id"])

        time.sleep(args.sleep)

    with open(args.config, "w") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, width=100)

    log.info("Done: enriched %d/%d episodes. Wrote %s", updated, len(episodes), args.config)


if __name__ == "__main__":
    main()