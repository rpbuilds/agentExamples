"""Answer a question by returning the chunks closest to it in vector space.

Retrieval is a single dot product against the whole matrix. At this corpus size that is
faster than any vector database and has no moving parts: the matrix row and the
chunks.jsonl line share an index, so a hit carries its own metadata.

Run from this folder, after prep.py and embed.py:
    python search.py "What was the Daniella Tahay settlement amount?"
    python search.py "kennel regulations" --k 3 --full
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from embed import DEFAULT_VECTORS_FILE, embed_query, load_index
from prep import DEFAULT_CHUNKS_FILE, load_chunks

DEFAULT_TOP_K: int = 5

# Characters of chunk text shown per hit unless --full is passed
_PREVIEW_CHARS: int = 400


class MinutesSearcher:
    """Holds the chunks, their vectors, and the model so repeated queries stay cheap.

    Loading the embedding model and the index takes seconds, so an object that keeps both
    in memory is what makes an interactive or multi-question session usable.
    """

    def __init__(
        self,
        chunks_file: Path = DEFAULT_CHUNKS_FILE,
        vectors_file: Path = DEFAULT_VECTORS_FILE,
    ) -> None:
        """Load the chunk metadata and matching vector matrix.

        Args:
            chunks_file: Path to chunks.jsonl.
            vectors_file: Path to embeddings.npy.

        Raises:
            ValueError: If the two files disagree on how many chunks there are, which
                means the index is stale and must be rebuilt.
        """
        self.chunks: list[dict[str, Any]] = load_chunks(chunks_file)
        self.vectors: np.ndarray = load_index(vectors_file)

        # A mismatch means prep.py ran without embed.py; silently searching would return
        # the wrong metadata for every hit
        if len(self.chunks) != self.vectors.shape[0]:
            raise ValueError(
                f"{len(self.chunks)} chunks but {self.vectors.shape[0]} vectors. "
                "Re-run: python embed.py --build"
            )

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Return the k chunks whose vectors sit closest to the query's.

        Args:
            query: The user's question in plain language.
            k: How many chunks to return.

        Returns:
            Chunk records sorted most similar first, each with an added "score" field
            holding cosine similarity in the range -1 to 1.
        """
        query_vector: np.ndarray = embed_query(query)

        # Vectors are unit length, so the dot product is cosine similarity
        scores: np.ndarray = self.vectors @ query_vector

        # argpartition finds the top k without sorting the whole corpus
        top_count: int = min(k, len(self.chunks))
        candidate_indices: np.ndarray = np.argpartition(-scores, top_count - 1)[:top_count]
        ranked_indices: np.ndarray = candidate_indices[np.argsort(-scores[candidate_indices])]

        results: list[dict[str, Any]] = []

        # Copy each record so the caller cannot mutate the in-memory corpus
        index: int
        for index in ranked_indices:
            result: dict[str, Any] = dict(self.chunks[int(index)])
            result["score"] = float(scores[int(index)])
            results.append(result)

        return results


def search(query: str, k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Run one query without managing a searcher, for quick scripts and tests.

    Args:
        query: The user's question.
        k: How many chunks to return.

    Returns:
        The top k chunk records with scores.
    """
    return MinutesSearcher().search(query, k)


def format_results(results: list[dict[str, Any]], full: bool = False) -> str:
    """Render results for a terminal, leading with the citation fields.

    Args:
        results: Output of MinutesSearcher.search.
        full: True to print whole chunks instead of a preview.

    Returns:
        A printable report.
    """
    lines: list[str] = []

    rank: int
    result: dict[str, Any]
    for rank, result in enumerate(results, start=1):
        item_label: str = (
            f"Item {result['item_number']}" if result["item_number"] is not None else "Meeting info"
        )
        pages: str = (
            f"p. {result['page_start']}"
            if result["page_start"] == result["page_end"]
            else f"pp. {result['page_start']}-{result['page_end']}"
        )

        lines.append(
            f"\n[{rank}] score {result['score']:.3f} | {result['meeting_date']} | "
            f"{item_label} | {result['section'].title()} | {pages}"
        )

        # Only show identifier fields that were actually found, to keep output readable
        identifiers: list[str] = []
        if result["ordinance_numbers"]:
            identifiers.append(f"Ordinance {', '.join(result['ordinance_numbers'])}")
        if result["resolution_numbers"]:
            identifiers.append(f"Resolution {', '.join(result['resolution_numbers'])}")
        if result["case_numbers"]:
            identifiers.append(f"Case {', '.join(result['case_numbers'])}")
        if result["districts"]:
            identifiers.append(f"District {', '.join(result['districts'])}")
        if result["vote"]:
            identifiers.append(f"Vote {result['vote']}")
        if result["outcome"]:
            identifiers.append(f"Outcome {result['outcome']}")
        if identifiers:
            lines.append("    " + " | ".join(identifiers))

        text: str = result["text"]
        if not full and len(text) > _PREVIEW_CHARS:
            text = f"{text[:_PREVIEW_CHARS]}..."
        lines.append(f"    {text}")

    return "\n".join(lines)


def main() -> None:
    """Search the minutes from the command line."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Semantic search over chunked City of Austin council minutes."
    )
    parser.add_argument("query", help="Your question, in quotes.")
    parser.add_argument(
        "--k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to return."
    )
    parser.add_argument("--full", action="store_true", help="Print whole chunks, not previews.")
    parser.add_argument(
        "--json", action="store_true", help="Print raw JSON results with all metadata."
    )
    parser.add_argument(
        "--chunks", type=Path, default=DEFAULT_CHUNKS_FILE, help="Path to chunks.jsonl."
    )
    parser.add_argument(
        "--vectors", type=Path, default=DEFAULT_VECTORS_FILE, help="Path to embeddings.npy."
    )
    args: argparse.Namespace = parser.parse_args()

    searcher: MinutesSearcher = MinutesSearcher(args.chunks, args.vectors)
    results: list[dict[str, Any]] = searcher.search(args.query, args.k)

    # JSON output exists so other scripts can consume retrieval without reparsing text
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"\nQuery: {args.query}")
        print(format_results(results, args.full))


if __name__ == "__main__":
    main()
