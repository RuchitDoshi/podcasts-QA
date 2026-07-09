"""
Embed chunks (data/chunks/*.json) using a local sentence-transformers model
and index them into a local ChromaDB collection for retrieval.

Runs entirely on CPU by default (bge-small is small enough that CPU is fine
for a corpus this size) -- no API cost, no GPU required, matches the
project's zero-cost stack.

Usage:
    python ingest/index.py --chunks data/chunks
    python ingest/index.py --chunks data/chunks --episode ep001 --overwrite
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("index")


def load_settings(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_embedder(settings: dict):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    e = settings["embeddings"]
    log.info("Loading embedding model %s (device=%s)", e["model_name"], e["device"])
    return SentenceTransformer(e["model_name"], device=e["device"])


def get_collection(settings: dict, reset: bool = False):
    try:
        import chromadb
    except ImportError:
        log.error("chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    chroma_dir = settings["paths"]["chroma_dir"]
    log.info("Connecting to ChromaDB at %s", chroma_dir)
    client = chromadb.PersistentClient(path=chroma_dir)

    collection_name = "podcast_chunks"
    if reset:
        try:
            client.delete_collection(collection_name)
            log.info("Deleted existing collection %s", collection_name)
        except Exception:
            pass  # collection didn't exist yet, nothing to delete

    return client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})


def flatten_metadata(record: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool -- flatten nested
    fields and stringify lists (tags, speakers) accordingly."""
    meta = record["metadata"]
    return {
        "episode_id": record["episode_id"],
        "start": record["start"],
        "end": record["end"],
        "title": meta.get("title") or "",
        "guest": meta.get("guest") or "",
        "date": meta.get("date") or "",
        "tags": ",".join(meta.get("tags") or []),
        "speakers": ",".join(meta.get("speakers") or []),
        "is_likely_ad": bool(meta.get("is_likely_ad", False)),
    }


def index_episode(
    collection, embedder, chunk_path: Path, batch_size: int = 32, include_ads: bool = False
) -> tuple[int, int]:
    with open(chunk_path) as f:
        records = json.load(f)

    if not records:
        log.warning("No chunks in %s -- skipping", chunk_path.name)
        return 0, 0

    skipped = 0
    if not include_ads:
        before = len(records)
        records = [r for r in records if not r["metadata"].get("is_likely_ad")]
        skipped = before - len(records)

    if not records:
        return 0, skipped

    ids = [r["chunk_id"] for r in records]
    texts = [r["text"] for r in records]
    metadatas = [flatten_metadata(r) for r in records]

    embeddings = embedder.encode(
        texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
    ).tolist()

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(records), skipped


def main():
    parser = argparse.ArgumentParser(description="Embed and index chunks into ChromaDB")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-chunks"),
    )
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--episode", type=str, default=None, help="Only index this episode id")
    parser.add_argument(
        "--overwrite", action="store_true", help="Reset the collection before indexing"
    )
    parser.add_argument(
        "--include-ads",
        action="store_true",
        help="Index chunks flagged as likely sponsor/ad reads too (excluded by default)",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    chunk_files = sorted(args.chunks.glob("*.json"))

    if args.episode:
        chunk_files = [f for f in chunk_files if f.stem == args.episode]
        if not chunk_files:
            log.error("No chunk file found for episode id %s", args.episode)
            sys.exit(1)

    if not chunk_files:
        log.warning("No chunk files found in %s -- run ingest/chunk.py first.", args.chunks)
        return

    embedder = get_embedder(settings)
    collection = get_collection(settings, reset=args.overwrite)

    total, total_skipped = 0, 0
    for chunk_path in chunk_files:
        n, skipped = index_episode(collection, embedder, chunk_path, include_ads=args.include_ads)
        log.info("Indexed %s: %d chunks (%d skipped as ads)", chunk_path.stem, n, skipped)
        total += n
        total_skipped += skipped

    log.info(
        "Done: %d episodes, %d chunks indexed, %d skipped as ads. Collection now has %d total items.",
        len(chunk_files),
        total,
        total_skipped,
        collection.count(),
    )


if __name__ == "__main__":
    main()
