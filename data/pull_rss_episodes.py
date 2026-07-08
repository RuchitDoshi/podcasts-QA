"""
Pull episode metadata from the Lex Fridman Podcast RSS feed and write
config/episodes.yaml automatically, instead of copying YouTube URLs by hand.

The feed gives direct MP3 enclosure URLs per episode, which download.py can
pass straight to yt-dlp (yt-dlp handles plain audio URLs fine, not just
YouTube/video platforms) -- no video download overhead, no diarization noise
from crowd/visual cues.

Usage:
    python data/pull_rss_episodes.py --limit 30
    python data/pull_rss_episodes.py --limit 30 --since 2025-01-01
    python data/pull_rss_episodes.py --limit 30 --tags-from-title
"""

import argparse
import html
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pull_rss_episodes")

FEED_URL = "https://lexfridman.com/feed/podcast/"

# Markers where Lex's boilerplate (sponsors, transcript link, contact info)
# reliably starts -- cut the description there instead of at a fixed length,
# so we keep the actual bio and drop the templated filler.
BOILERPLATE_MARKERS = [
    "Thank you for listening",
    "See below for timestamps",
    "CONTACT LEX",
    "PODCAST LINKS",
    "Transcript:",
]


def clean_description(raw_html: str, max_len: int = 500) -> str:
    if not raw_html:
        return ""

    text = html.unescape(raw_html)
    text = re.sub(r"<br\s*/?>", "\n", text)          # line breaks before stripping tags
    text = re.sub(r"<[^>]+>", "", text)               # strip remaining HTML tags
    text = re.sub(r"[ \t]+", " ", text).strip()

    cut_at = len(text)
    for marker in BOILERPLATE_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)

    text = text[:cut_at].strip()
    return text[:max_len].strip()


def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:max_len]


def guest_from_title(title: str) -> str:
    # Rough fallback only -- breaks on multi-guest titles, question-style
    # titles, etc. Real guest extraction happens in enrich_episodes.py using
    # the RSS description, which states the guest explicitly far more often.
    cleaned = re.sub(r"^#?\d+\s*[-–—]\s*", "", title)
    return cleaned.split(":")[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Build episodes.yaml from the Lex Fridman RSS feed")
    parser.add_argument("--feed-url", type=str, default=FEED_URL)
    parser.add_argument("--limit", type=int, default=30, help="Max number of episodes to include")
    parser.add_argument("--since", type=str, default=None, help="Only include episodes after this date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, default=Path("config/episodes.yaml"))
    parser.add_argument("--tags-from-title", action="store_true",
                         help="Naively derive tags from title keywords (rough — expect to hand-edit after)")
    args = parser.parse_args()

    try:
        import feedparser
    except ImportError:
        log.error("feedparser is not installed. Run: pip install feedparser")
        sys.exit(1)

    log.info("Fetching %s", args.feed_url)
    feed = feedparser.parse(args.feed_url)

    if feed.bozo and not feed.entries:
        log.error("Failed to parse feed: %s", feed.bozo_exception)
        sys.exit(1)

    since_dt = datetime.strptime(args.since, "%Y-%m-%d") if args.since else None

    episodes = []
    for entry in feed.entries:
        published = entry.get("published_parsed")
        pub_date = datetime(*published[:6]) if published else None

        if since_dt and pub_date and pub_date < since_dt:
            continue

        enclosure_url = None
        for link in entry.get("links", []):
            if link.get("rel") == "enclosure" or "audio" in link.get("type", ""):
                enclosure_url = link.get("href")
                break

        if not enclosure_url:
            log.warning("No audio enclosure found for entry %r — skipping", entry.get("title"))
            continue

        title = entry.get("title", "untitled")
        episode_id = slugify(title) or f"ep_{len(episodes)+1:03d}"
        description = clean_description(entry.get("summary") or "")

        episode = {
            "id": episode_id,
            "url": enclosure_url,
            "title": title,
            "guest": guest_from_title(title),  # rough fallback -- run enrich_episodes.py after
            "date": pub_date.strftime("%Y-%m-%d") if pub_date else "unknown",
            "description": description,  # cleaned + trimmed to the bio, boilerplate stripped
            "tags": [],
        }

        if args.tags_from_title:
            # rough keyword pass -- meant as a starting point, not final tags
            lowered = title.lower()
            for kw in ["ai", "physics", "history", "space", "robot", "brain", "philosophy",
                       "math", "biology", "economics", "war", "music"]:
                if kw in lowered:
                    episode["tags"].append(kw)

        episodes.append(episode)

        if len(episodes) >= args.limit:
            break

    if not episodes:
        log.warning("No episodes matched the given filters — nothing written.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump({"episodes": episodes}, f, sort_keys=False, allow_unicode=True, width=100)

    log.info("Wrote %d episodes to %s", len(episodes), args.output)
    log.info("Guest names are a rough title-parse fallback -- run "
              "data/enrich_episodes.py to fix guests and generate tags from "
              "the episode descriptions before running data/download.py.")


if __name__ == "__main__":
    main()