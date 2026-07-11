"""
Checks every chunk in the real corpus against bge-small-en-v1.5's max
sequence length (512 tokens). Chunks over that limit get silently truncated
by sentence-transformers during embedding -- no error, no warning, just
quiet data loss in whatever text falls past token 512. This was named as an
unverified risk early in the project (chunking targets ~60-120s of dense
speech, which for a fast talker could plausibly exceed it) and never
actually checked against real data until now.

Uses the SAME tokenizer the embedding model actually uses internally
(loaded via the sentence-transformers model object, not a separate
from-scratch AutoTokenizer load) so the counts reported here are exactly
what the embedding step itself would see -- not an approximation.

Usage:
    python eval/check_token_lengths.py
    python eval/check_token_lengths.py --chunks data/chunks --limit 512
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieve.hybrid import get_embedder, load_corpus  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("check_token_lengths")


def count_tokens(embedder, text: str) -> int:
    """Uses the embedding model's own tokenizer, matching exactly what
    happens internally during .encode() -- not a separate/approximate count."""
    tokenizer = embedder.tokenizer
    return len(tokenizer.encode(text, add_special_tokens=True))


def main():
    parser = argparse.ArgumentParser(
        description="Check corpus chunks against the embedding model's max sequence length"
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/podcast-audio-chunks"),
    )
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument(
        "--limit", type=int, default=512, help="Token limit to check against (bge-small's max)"
    )
    parser.add_argument(
        "--warn-margin",
        type=int,
        default=30,
        help="Also flag chunks within this many tokens of the limit",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        settings = yaml.safe_load(f)

    corpus = load_corpus(args.chunks)
    if not corpus:
        log.error("No chunks found in %s -- run ingest/chunk.py first.", args.chunks)
        sys.exit(1)

    log.info("Loading embedder (using its actual tokenizer, not an approximation)...")
    embedder = get_embedder(settings)

    log.info("Checking %d chunks against a %d token limit...", len(corpus), args.limit)

    over_limit = []
    near_limit = []
    all_counts = []

    for record in corpus:
        n_tokens = count_tokens(embedder, record["text"])
        all_counts.append(n_tokens)

        if n_tokens > args.limit:
            over_limit.append((record["chunk_id"], n_tokens))
        elif n_tokens > args.limit - args.warn_margin:
            near_limit.append((record["chunk_id"], n_tokens))

    print("\n" + "=" * 70)
    print(f"TOKEN LENGTH CHECK -- bge-small max sequence length = {args.limit}")
    print("=" * 70)
    print(f"Total chunks checked: {len(all_counts)}")
    print(f"Max token count seen: {max(all_counts)}")
    print(f"Avg token count:      {sum(all_counts) / len(all_counts):.1f}")

    if over_limit:
        print(
            f"\nOVER LIMIT ({len(over_limit)} chunks) -- these are being silently truncated during embedding:"
        )
        for chunk_id, n in sorted(over_limit, key=lambda x: -x[1])[:20]:
            print(f"  {n:>4} tokens  {chunk_id}")
        if len(over_limit) > 20:
            print(f"  ... and {len(over_limit) - 20} more")
    else:
        print(
            "\nNo chunks exceed the limit -- no silent truncation happening in the current corpus."
        )

    if near_limit:
        print(
            f"\nNEAR LIMIT ({len(near_limit)} chunks, within {args.warn_margin} tokens) -- worth watching if "
            "chunking parameters change:"
        )
        for chunk_id, n in sorted(near_limit, key=lambda x: -x[1])[:10]:
            print(f"  {n:>4} tokens  {chunk_id}")

    if over_limit:
        print(
            f"\nRecommendation: {len(over_limit)}/{len(all_counts)} chunks "
            f"({len(over_limit) / len(all_counts):.1%}) are losing content silently. Consider lowering "
            "chunking.max_seconds in settings.yaml, or splitting long chunks specifically for embedding "
            "while keeping the original text for display/citation purposes."
        )


if __name__ == "__main__":
    main()
