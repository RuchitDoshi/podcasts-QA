import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        description="Query the ChromaDB index for podcast chunks"
    )
    argument_parser.add_argument(
        "--query",
        type=str,
        default="What does Kaldellis say about the fall of Constantinople?",
        help="Query string to search for in the indexed chunks",
    )
    argument_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Path to the configuration file",
    )
    argument_parser.add_argument(
        "--chroma-db-path",
        type=Path,
        default=Path("/home/ruchit/ruchit-personal/datasets/chroma_db"),
        help="Path to the ChromaDB database directory",
    )
    argument_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top results to retrieve"
    )
    args = argument_parser.parse_args()


client = chromadb.PersistentClient(path=args.chroma_db_path)
collection = client.get_collection("podcast_chunks")
print("Total chunks in index:", collection.count())

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
query = args.query
query_emb = embedder.encode([query], normalize_embeddings=True).tolist()

results = collection.query(query_embeddings=query_emb, n_results=args.top_k)
for i, (doc, _) in enumerate(zip(results["documents"][0], results["metadatas"][0], strict=False)):
    print(f"Result {i+1}:")
    print(doc[:200])
    print()
