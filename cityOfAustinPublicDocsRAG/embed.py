"""Embed chunks and queries into the same vector space, and store the vectors.

The embedder is a local sentence-transformers model, so no API key is needed and the
documents never leave the machine. Vectors are normalised to unit length at encode time,
which turns cosine similarity into a plain dot product in search.py.

Run from this folder, after prep.py:
    python embed.py --build
    python embed.py --stats
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from prep import DEFAULT_CHUNKS_FILE, embedding_text, load_chunks

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Small, fast, CPU-friendly, and strong on short-passage retrieval. 384 dimensions,
# 512-token input limit.
MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
MODEL_MAX_TOKENS: int = 512

# bge models expect this prefix on queries but not on documents
QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

DEFAULT_VECTORS_FILE: Path = Path(__file__).resolve().parent / "index" / "embeddings.npy"

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


# ---------------------------------------------------------------------------
# PRIVATE HELPERS
# ---------------------------------------------------------------------------


def _get_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the embedding model once per process, since loading costs seconds.

    Args:
        model_name: Hugging Face model id.

    Returns:
        A ready SentenceTransformer, cached by name.
    """
    if model_name not in _MODEL_CACHE:
        print(f"Loading embedding model {model_name} (first run downloads it)")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Embed document texts as unit vectors.

    Args:
        texts: Strings to embed, already carrying their context headers.
        model_name: Hugging Face model id.

    Returns:
        Array of shape (len(texts), dimension), each row L2-normalised.
    """
    model: SentenceTransformer = _get_model(model_name)
    vectors: np.ndarray = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=len(texts) > 32
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str, model_name: str = MODEL_NAME) -> np.ndarray:
    """Embed a user question into the same space as the chunks.

    The bge query prefix is added here and only here, so callers cannot forget it and
    silently lose retrieval quality.

    Args:
        query: The user's question.
        model_name: Hugging Face model id.

    Returns:
        A single unit vector of shape (dimension,).
    """
    model: SentenceTransformer = _get_model(model_name)
    vector: np.ndarray = model.encode(
        f"{QUERY_PREFIX}{query}", normalize_embeddings=True
    )
    return np.asarray(vector, dtype=np.float32)


def token_length(text: str, model_name: str = MODEL_NAME) -> int:
    """Count tokens with the embedder's own tokenizer, for checking the 512-token limit.

    Args:
        text: Text to measure.
        model_name: Hugging Face model id.

    Returns:
        Token count including the model's special tokens.
    """
    model: SentenceTransformer = _get_model(model_name)
    return len(model.tokenizer.encode(text))


def build_index(
    chunks_file: Path = DEFAULT_CHUNKS_FILE,
    vectors_file: Path = DEFAULT_VECTORS_FILE,
    model_name: str = MODEL_NAME,
) -> np.ndarray:
    """Embed every chunk and save the matrix next to chunks.jsonl.

    Row order is the chunk file's line order, which is the only link between a vector and
    its metadata. Nothing may reorder one without the other.

    Args:
        chunks_file: Path to chunks.jsonl from prep.py.
        vectors_file: Destination .npy path.
        model_name: Hugging Face model id.

    Returns:
        The saved embedding matrix.
    """
    chunks: list[dict[str, Any]] = load_chunks(chunks_file)
    texts: list[str] = [embedding_text(chunk) for chunk in chunks]

    vectors: np.ndarray = embed_texts(texts, model_name)

    vectors_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_file, vectors)

    print(f"Embedded {vectors.shape[0]} chunks into {vectors.shape[1]} dimensions")
    print(f"Wrote vectors to {vectors_file}")
    return vectors


def load_index(vectors_file: Path = DEFAULT_VECTORS_FILE) -> np.ndarray:
    """Load the embedding matrix built by build_index.

    Args:
        vectors_file: Path to embeddings.npy.

    Returns:
        The embedding matrix.

    Raises:
        FileNotFoundError: If the index has not been built yet.
    """
    if not vectors_file.exists():
        raise FileNotFoundError(f"{vectors_file} not found. Run: python embed.py --build")
    return np.load(vectors_file)


def report_token_stats(
    chunks_file: Path = DEFAULT_CHUNKS_FILE, model_name: str = MODEL_NAME
) -> dict[str, Any]:
    """Report real token counts per chunk, to show how much headroom the 512 limit leaves.

    This answers a planning question directly: whether chunks fit the embedder today, and
    how many tokens a retrieved set would add to a future generation prompt.

    Args:
        chunks_file: Path to chunks.jsonl.
        model_name: Hugging Face model id.

    Returns:
        Summary statistics, also printed to stdout.
    """
    chunks: list[dict[str, Any]] = load_chunks(chunks_file)
    counts: list[int] = [token_length(embedding_text(chunk), model_name) for chunk in chunks]
    sorted_counts: list[int] = sorted(counts)

    over_limit: list[dict[str, Any]] = [
        chunk for chunk, count in zip(chunks, counts) if count > MODEL_MAX_TOKENS
    ]

    stats: dict[str, Any] = {
        "chunks": len(counts),
        "total_tokens": sum(counts),
        "min": sorted_counts[0],
        "median": sorted_counts[len(sorted_counts) // 2],
        "p95": sorted_counts[int(len(sorted_counts) * 0.95)],
        "max": sorted_counts[-1],
        "mean": round(sum(counts) / len(counts), 1),
        "over_512": len(over_limit),
    }

    print(f"Chunks:            {stats['chunks']}")
    print(f"Tokens per chunk:  min {stats['min']}, median {stats['median']}, "
          f"mean {stats['mean']}, p95 {stats['p95']}, max {stats['max']}")
    print(f"Corpus total:      {stats['total_tokens']} tokens")
    print(f"Over {MODEL_MAX_TOKENS}-token limit: {stats['over_512']} chunk(s)")
    print(f"Top-5 retrieval would add roughly {stats['p95'] * 5} tokens to a prompt")

    # Name the offenders so the chunk size in prep.py can be tuned against real cases
    chunk: dict[str, Any]
    for chunk in over_limit[:5]:
        print(f"  truncated: {chunk['chunk_id']}")

    return stats


def main() -> None:
    """Build the vector index or report token statistics from the command line."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Embed council-minutes chunks into a numpy vector index."
    )
    parser.add_argument("--build", action="store_true", help="Embed chunks and save the index.")
    parser.add_argument(
        "--stats", action="store_true", help="Print real token counts per chunk."
    )
    parser.add_argument(
        "--chunks", type=Path, default=DEFAULT_CHUNKS_FILE, help="Path to chunks.jsonl."
    )
    parser.add_argument(
        "--vectors", type=Path, default=DEFAULT_VECTORS_FILE, help="Path to embeddings.npy."
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Hugging Face model id.")
    args: argparse.Namespace = parser.parse_args()

    # Default to building, since that is why this script is usually run
    if args.stats:
        report_token_stats(args.chunks, args.model)
    if args.build or not args.stats:
        build_index(args.chunks, args.vectors, args.model)


if __name__ == "__main__":
    main()
