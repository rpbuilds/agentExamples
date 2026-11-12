"""Measure whether the right agenda item lands in the top k for known questions.

Retrieval quality is otherwise invisible: results always look plausible. These cases have
answers verified by hand in the PDFs, so a change to chunking, the embedder, or the
context header can be judged instead of guessed at. Uses no test framework, only argparse.

Run from this folder, after prep.py and embed.py:
    python test_retrieval.py
    python test_retrieval.py --k 1 --verbose
"""

import argparse
from dataclasses import dataclass
from typing import Any

from search import DEFAULT_TOP_K, MinutesSearcher


@dataclass(frozen=True)
class RetrievalCase:
    """One question with the chunk that should answer it, verified by hand in the PDF.

    Attributes:
        question: Asked the way a person would ask it, not using the document's wording.
        meeting_date: ISO date of the meeting holding the answer.
        item_number: Agenda item number holding the answer.
        note: Why this case is worth testing.
    """

    question: str
    meeting_date: str
    item_number: int
    note: str


# ---------------------------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------------------------

CASES: list[RetrievalCase] = [
    RetrievalCase(
        question="How much did the city pay to settle the Daniella Tahay lawsuit?",
        meeting_date="2026-10-22",
        item_number=4,
        note="Dollar figure sits in a sentence after the vote, not in the item text.",
    ),
    RetrievalCase(
        question="What was the settlement amount in the Elena Torres case?",
        meeting_date="2026-10-22",
        item_number=5,
        note="Near-duplicate of the Tahay item, so it tests discrimination between them.",
    ),
    RetrievalCase(
        question="Did council pass any rules about dog kennels and escaping animals?",
        meeting_date="2026-10-22",
        item_number=6,
        note="Everyday phrasing against legal wording ('kennel operators', 'escape-proof').",
    ),
    RetrievalCase(
        question="What new advisory board was created for the Millennium Youth complex?",
        meeting_date="2026-10-22",
        item_number=3,
        note="Named entity that appears only once in the corpus.",
    ),
    RetrievalCase(
        question="Was Samsung nominated for a state enterprise project designation?",
        meeting_date="2025-11-20",
        item_number=10,
        note="Company name plus a state program name.",
    ),
    RetrievalCase(
        question="How much money went to Balance Dance Studios?",
        meeting_date="2025-11-20",
        item_number=11,
        note="Economic development agreement; one of many similar funding items.",
    ),
    RetrievalCase(
        question="What did council approve for the Longhorn Dam public art project?",
        meeting_date="2025-11-20",
        item_number=3,
        note="Art in Public Places contract amendment with an artist by name.",
    ),
    RetrievalCase(
        question="How is the police radio modernization project being paid for?",
        meeting_date="2025-11-20",
        item_number=7,
        note="Question asks about funding mechanism, which is tax-exempt GO debt.",
    ),
    RetrievalCase(
        question="Which department now oversees the People's Renaissance Market?",
        meeting_date="2025-11-20",
        item_number=2,
        note="Answer is a department transfer buried in a code amendment.",
    ),
    RetrievalCase(
        question="Did the city lower the property tax rate after the election?",
        meeting_date="2025-11-20",
        item_number=6,
        note="Appears late in the file where the PDF's text layout degrades.",
    ),
    RetrievalCase(
        question="What support was given to Esther's Follies?",
        meeting_date="2026-10-22",
        item_number=8,
        note="Item number was OCR'd as '8,' instead of '8.', testing the tolerant splitter.",
    ),
    RetrievalCase(
        question="What happened with the No Wrong Door initiative reporting?",
        meeting_date="2026-10-22",
        item_number=45,
        note="Came from a council committee, so it sits outside the consent agenda.",
    ),
    RetrievalCase(
        question="Was there an eminent domain action for flood risk near Barton Creek?",
        meeting_date="2026-10-22",
        item_number=47,
        note="Long legal description, so the answer may live in a later sub-chunk.",
    ),
    RetrievalCase(
        question="What was decided about rezoning on Dalton Lane near the airport?",
        meeting_date="2026-10-22",
        item_number=51,
        note="Zoning case C14-2026-0019; tests the last item in the file.",
    ),
    RetrievalCase(
        question="How much in road bonds did Pilot Knob Municipal Utility District No. 5 get?",
        meeting_date="2026-10-22",
        item_number=18,
        note=(
            "Three near-identical Pilot Knob items differ only by district number, so "
            "embeddings score them within 0.003 of each other. This is the case that "
            "argues for adding keyword search, and the item whose number the OCR mangled."
        ),
    ),
]


# ---------------------------------------------------------------------------
# PRIVATE HELPERS
# ---------------------------------------------------------------------------


def _matches(result: dict[str, Any], case: RetrievalCase) -> bool:
    """Decide whether a single hit is the chunk the case expects.

    Args:
        result: One search result.
        case: The case being checked.

    Returns:
        True when the hit is any sub-chunk or occurrence of the expected item.
    """
    return (
        result["meeting_date"] == case.meeting_date
        and result["item_number"] == case.item_number
    )


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def run_cases(k: int = DEFAULT_TOP_K, verbose: bool = False) -> dict[str, Any]:
    """Run every case and report how often the expected item was retrieved.

    Args:
        k: How many chunks to retrieve per question.
        verbose: True to print the ranked items for each case.

    Returns:
        Summary with pass count, total, recall, and the failing cases.
    """
    searcher: MinutesSearcher = MinutesSearcher()
    passed: int = 0
    failures: list[tuple[RetrievalCase, list[str]]] = []
    ranks: list[int] = []

    # One question at a time; the searcher keeps the model loaded between them
    case: RetrievalCase
    for case in CASES:
        results: list[dict[str, Any]] = searcher.search(case.question, k)
        hit_rank: int | None = next(
            (rank for rank, result in enumerate(results, start=1) if _matches(result, case)),
            None,
        )
        retrieved: list[str] = [
            f"{result['meeting_date']} item {result['item_number']} ({result['score']:.3f})"
            for result in results
        ]

        if hit_rank is not None:
            passed += 1
            ranks.append(hit_rank)
            print(f"PASS rank {hit_rank}  {case.question}")
        else:
            failures.append((case, retrieved))
            print(f"FAIL          {case.question}")
            print(f"    expected {case.meeting_date} item {case.item_number}")
            print(f"    got      {', '.join(retrieved)}")

        if verbose and hit_rank is not None:
            print(f"    got      {', '.join(retrieved)}")

    recall: float = passed / len(CASES)
    mean_rank: float = sum(ranks) / len(ranks) if ranks else 0.0

    print(f"\n{passed}/{len(CASES)} cases found the expected item in the top {k} "
          f"(recall@{k} = {recall:.0%})")
    if ranks:
        print(f"Mean rank of the correct chunk when found: {mean_rank:.1f}")

    # Restate the notes for failures, since they say what the hard part was
    failed_case: RetrievalCase
    for failed_case, _ in failures:
        print(f"  missed: {failed_case.note}")

    return {
        "passed": passed,
        "total": len(CASES),
        "recall": recall,
        "mean_rank": mean_rank,
        "failures": [case.question for case, _ in failures],
    }


def main() -> None:
    """Run the retrieval cases from the command line."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Check that known questions retrieve the right agenda item."
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_TOP_K, help="Chunks to retrieve per question."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show retrieved items for passing cases too."
    )
    args: argparse.Namespace = parser.parse_args()

    run_cases(args.k, args.verbose)


if __name__ == "__main__":
    main()
