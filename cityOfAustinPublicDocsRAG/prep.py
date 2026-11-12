"""Turn City of Austin council minutes PDFs into retrievable, item-level chunks.

Each Austin City Council agenda item is a self-contained unit of meaning: what was
proposed, how it was funded, and how Council voted. This module splits the minutes on
item boundaries so that one chunk answers one question, then attaches the metadata a
caller needs to cite the answer (meeting date, section, item number, page range,
ordinance/resolution/case numbers, districts, vote, outcome).

Run from this folder to write index/chunks.jsonl:
    python prep.py --docs coaDocs --out index/chunks.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pymupdf

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DEFAULT_DOCS_DIR: Path = Path(__file__).resolve().parent / "coaDocs"
DEFAULT_CHUNKS_FILE: Path = Path(__file__).resolve().parent / "index" / "chunks.jsonl"

# Highest plausible agenda item number; anything larger is OCR or wrapped text
_MAX_ITEM_NUMBER: int = 200

# Sub-chunk sizing. bge-small-en-v1.5 truncates at 512 tokens, so chunks are capped below
# that with room left for the context header prepended at embed time. The ratio is high
# because these minutes are dense with case numbers, hyphenated zoning codes, and dollar
# figures, which cost several tokens per word; embed.py --stats checks the real counts.
_MAX_CHUNK_TOKENS: int = 440
_TOKENS_PER_WORD: float = 1.8

# Replacement character produced where the scan defeated the OCR engine
_BAD_CHAR: str = "�"

# Characters the PDF text layer uses for line-break hyphens: soft hyphen, plain hyphen,
# and Unicode hyphen. The hyphen is part of the text ("Harper-Madison"), never a wrap.
_SOFT_HYPHENS: tuple[str, ...] = ("­", "-", "‐")

# Closing quotes and brackets that may sit after the period ending a sentence
_TRAILING_PUNCTUATION: str = "”’\"')] "

# Words a number continues rather than follows, which reject the "2." left behind by a
# wrapped "District(s) Affected: District 2.". These only disqualify a candidate when the
# previous line did not finish a sentence, because "...Homestead Preservation Districts."
# is a real sentence end with a real item after it.
_NUMBER_PRECEDING_WORDS: frozenset[str] = frozenset(
    {
        "abstract", "article", "block", "chapter", "district", "districts", "item",
        "items", "lot", "lots", "phase", "precinct", "section", "series", "survey",
        "tract", "unit",
    }
)

# Abbreviations whose own period makes a wrapped line look like a finished sentence, as in
# "Abstract No." followed by "2, Travis County". A number after one is never an item.
_ABBREVIATIONS_BEFORE_NUMBER: frozenset[str] = frozenset({"no", "nos"})

# Outcome keywords, checked in order so that the most specific wins
_OUTCOME_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("postponed indefinitely", "postponed_indefinitely"),
    ("was postponed", "postponed"),
    ("was continued", "continued"),
    ("withdrawn", "withdrawn"),
    ("has been removed", "removed"),
    ("was denied", "denied"),
    ("failed", "failed"),
    ("approved on consent", "approved_on_consent"),
    ("was approved", "approved"),
)

# ---------------------------------------------------------------------------
# REGEXES: CLEANING THE PDF TEXT
# ---------------------------------------------------------------------------

# Running header stamped on every page.
#   "REGULAR COUNCIL MINUTES THURSDAY, NOVEMBER 20, 2025"  -> match
#   "SPECIAL CALLED MEETING MINUTES"                       -> match
#   "Approve the minutes of the prior meeting"             -> no match
_PAGE_HEADER_RE = re.compile(
    r"""
    ^ \s*
    (REGULAR | SPECIAL | WORK[ ]SESSION | BUDGET)   # kind of meeting
    [A-Z ]*                                         # filler like "CALLED " or "COUNCIL "
    (COUNCIL[ ])?
    MINUTES \b .*                                   # rest of the header is ignored
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

# A line holding nothing but a page number.
#   "14"  -> match        "14."  -> no match        "2025"  -> no match
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")

# The 2025 file splits its running header across two lines, leaving the meeting date on a
# line of its own. In all caps it would otherwise be mistaken for a section banner.
#   "THURSDAY, NOVEMBER 20, 2025"  -> match
#   "THURSDAY, NOVEMBER20, 2025"   -> match (the text layer drops spaces)
_PAGE_DATE_RE = re.compile(
    r"""
    ^ \s*
    (MON | TUES | WEDNES | THURS | FRI | SATUR | SUN) DAY ,  \s+   # "THURSDAY,"
    [A-Z]+ \s*                                                     # month name
    \d{1,2} \s* ,? \s*                                             # day of month
    \d{4}                                                          # year
    \s* $
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# REGEXES: FINDING ITEM BOUNDARIES
# ---------------------------------------------------------------------------

# Section banner: a short all-caps line.
#   "CONSENT AGENDA"                                     -> match
#   "ZONING AND NEIGHBORHOOD PLAN AMENDMENTS"            -> match
#   "Approve a contract with Acme Inc."                  -> no match (lowercase)
_SECTION_RE = re.compile(
    r"""
    ^ [A-Z]                         # starts with a capital
    [A-Z0-9 ,.&/()'\-]{5,}          # then at least 5 more caps, digits, or banner punctuation
    $
    """,
    re.VERBOSE,
)

# Start of an agenda item. The number normally sits alone on its own line; the same-line
# form is accepted too. The 2026 file is a scan, so OCR noise is tolerated.
#   "22."              -> number 22, rest ""
#   "] 8."             -> junk "]", number 8, rest ""      (OCR ate the leading 1 of 18)
#   "7, Approve a..."  -> number 7, rest "Approve a..."    (comma instead of period)
#   "District 2."      -> no match (starts with a word)
_ITEM_START_RE = re.compile(
    r"""
    ^
    (?P<junk>   [\]\[|}{>]{0,2} )   # OCR debris sitting where a digit should be
    \s*
    (?P<number> \d{1,3} )           # the agenda item number
    \s* [.,] \s*                    # its period, or a comma from a bad scan
    (?P<rest>   .* )                # item text, on the rare lines that carry it
    $
    """,
    re.VERBOSE,
)

# ---------------------------------------------------------------------------
# REGEXES: JOINING LINES AND SPLITTING LONG ITEMS
# ---------------------------------------------------------------------------

# The 2025 text layer scatters a space on one side of a hyphen. A space on only one side
# is always this artifact, while a space on both sides is a real dash, so only the
# one-sided forms are closed up. Each entry is (pattern, replacement).
#   "Chapter 14- 2"           -> "Chapter 14-2"
#   "20251120 -002"           -> "20251120-002"
#   "Proclamation - Transgender Day" is left alone
_SPLIT_HYPHEN_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(\w)-\s+(\w)"), r"\1-\2"),  # space after the hyphen
    (re.compile(r"(\w)\s+-(\w)"), r"\1-\2"),  # space before the hyphen
)

# Where a long item may be cut: after sentence punctuation, before a capital or "(".
#   "...approved. The motion..."  -> cut between "approved." and "The"
#   "...Lot 4. 5 acres..."        -> no cut (next word is not a capital)
_SENTENCE_SPLIT_RE = re.compile(
    r"""
    (?<= [.;:] )        # preceded by sentence punctuation
    \s+
    (?= [A-Z(] )        # followed by the start of a new sentence
    """,
    re.VERBOSE,
)

# ---------------------------------------------------------------------------
# REGEXES: METADATA
# ---------------------------------------------------------------------------

# Numbers are OCR'd with stray spaces, so these capture them loosely; the caller strips
# the whitespace.
#   "Ordinance No. 20251120-008"   -> "20251120-008"
#   "Ordinance No. 20251120 -002"  -> "20251120 -002"
_ORDINANCE_RE = re.compile(
    r"""
    Ordinance[ ]No \.? \s*
    (?P<number> [\d\-\s]{8,20} \d )     # digits, hyphens and stray spaces, ending on a digit
    """,
    re.IGNORECASE | re.VERBOSE,
)

#   "Resolution No. 20251120-050"  -> "20251120-050"
_RESOLUTION_RE = re.compile(
    r"""
    Resolution[ ]No \.? \s*
    (?P<number> [\d\-\s]{8,20} \d )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Zoning/plat case numbers and district-court cause numbers.
#   "C14-2024-0123"        -> zoning case
#   "C14-2024-0123.02"     -> zoning case with a suffix
#   "D-1-GN-23-002694"     -> district court cause number
_CASE_RE = re.compile(
    r"""
    \b
    (?P<case>
        C \d{2} - \d{4} - \d{3,4} (?: \. \d+ )?     # C14-2024-0123, optional ".02"
        |
        D - \d - [A-Z]{2} - \d{2} - \d{6}           # D-1-GN-23-002694
    )
    \b
    """,
    re.VERBOSE,
)

# The district number from a "District(s) Affected" line.
#   "District(s) Affected: District 2"  -> "2"
#   "District Affected District 10"     -> "10"
_DISTRICT_RE = re.compile(
    r"""
    District \(? s? \)? \s*     # "District", "District(s)", or "Districts"
    (?: Affected :? )? \s*      # optional "Affected:"
    District \s*
    (?P<number> \d{1,2} )
    """,
    re.VERBOSE,
)

# Dollar amounts, with or without cents.
#   "$1,250,000"  -> match        "$45.50"  -> match
_DOLLAR_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")

# The recorded vote.
#   "approved on an 11-0 vote"  -> yes 11, no 0
#   "passed on a 9 - 2 vote"    -> yes 9, no 2
_VOTE_RE = re.compile(
    r"""
    \b
    (?P<yes> \d{1,2} ) \s* - \s*
    (?P<no>  \d{1,2} ) \s+
    vote \b
    """,
    re.VERBOSE,
)

# The meeting date as it appears in the filename.
#   "20251120, Minutes"  -> year 2025, month 11, day 20
_DATE_IN_NAME_RE = re.compile(r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})")

# ---------------------------------------------------------------------------
# PDF TEXT CLEANUP
# ---------------------------------------------------------------------------


def _meeting_date_from_name(pdf_path: Path) -> str:
    """ISO date from the filename ("20251120, Minutes.PDF"), which beats OCR'd text."""
    match = _DATE_IN_NAME_RE.search(pdf_path.stem)
    if match is None:
        return ""
    return f"{match['year']}-{match['month']}-{match['day']}"


def _extract_lines(pdf_path: Path) -> list[tuple[str, int]]:
    """Read a PDF as (line, page) pairs, minus running headers, page numbers, and blanks.

    Page numbers stay attached to each line because items routinely straddle a page break
    and a citation is only useful if it names the right page.
    """
    lines: list[tuple[str, int]] = []
    document = pymupdf.open(pdf_path)

    try:
        for page_index in range(document.page_count):
            raw_text = document.load_page(page_index).get_text("text")

            for raw_line in raw_text.splitlines():
                line = " ".join(raw_line.replace(_BAD_CHAR, "'").split())

                # Boilerplate would otherwise land in the middle of an item
                is_boilerplate = (
                    _PAGE_HEADER_RE.match(line)
                    or _PAGE_NUMBER_RE.match(line)
                    or _PAGE_DATE_RE.match(line)
                )
                if line and not is_boilerplate:
                    lines.append((line, page_index + 1))
    finally:
        document.close()

    return lines


def _join_lines(lines: list[str]) -> str:
    """Join wrapped PDF lines into prose, keeping hyphens that a line break split.

    Both files break lines mid-token on a hyphen (plain in 2025, soft in the 2026 scan),
    and in every case found the hyphen belongs to the text ("Harper-Madison",
    "D-1-GN-23-002694"). So the hyphen is kept and the halves are joined with no space.
    """
    parts: list[str] = []

    for line in lines:
        # A line ending on a hyphen runs straight into the next one
        if parts and line and parts[-1].endswith(_SOFT_HYPHENS):
            parts[-1] = parts[-1].rstrip("".join(_SOFT_HYPHENS)) + "-" + line
        else:
            parts.append(line)

    text = " ".join(parts)

    for pattern, replacement in _SPLIT_HYPHEN_FIXES:
        text = pattern.sub(replacement, text)

    return text


# ---------------------------------------------------------------------------
# SPOTTING ITEM STARTS AND SECTION BANNERS
# ---------------------------------------------------------------------------


def _is_section_header(line: str) -> bool:
    """True for a short all-caps banner with no sentence punctuation inside it."""
    return (
        _SECTION_RE.match(line) is not None
        and len(line) <= 90
        and "." not in line.rstrip(".")
    )


def _is_unfinished_header(header: str) -> bool:
    """True when a banner has an unclosed parenthesis or trails off on AND/OR."""
    return header.count("(") > header.count(")") or header.rstrip().endswith((" AND", " OR"))


def _ends_a_sentence(line: str) -> bool:
    """True when a line finishes a sentence, allowing for a closing quote mark."""
    return line.rstrip(_TRAILING_PUNCTUATION).endswith((".", ":", "!", "?"))


def _last_word(line: str) -> str:
    """Final word of a line, lowercased and stripped of punctuation ("" if empty)."""
    words = line.split()
    if not words:
        return ""
    return words[-1].strip(".,:;()“”‘’\"'").lower()


def _repair_item_number(number: int, junk: str, last_number: int | None) -> int:
    """Recover a leading digit the OCR read as a bracket, as in "] 8." for item 18.

    Deliberately narrow: it fires only when junk was stripped off the front and prefixing
    a "1" gives exactly the next number in sequence, so it cannot renumber items that are
    merely out of order.
    """
    if not junk or last_number is None:
        return number

    candidate = int(f"1{number}")
    if candidate <= _MAX_ITEM_NUMBER and candidate == last_number + 1:
        return candidate
    return number


def _is_item_start(line: str, previous_line: str) -> tuple[int, str, str] | None:
    """Decide whether a line opens a new agenda item, rejecting wrapped text.

    Two false positives are rejected. A bare "2." may be the tail of "District(s)
    Affected: District 2.", caught by the word the previous line ends on. A line like
    "1, Amended Plat of Lots 1-6" is a wrapped legal description, caught by requiring a
    finished sentence before an item whose text starts on the same line.

    Returns:
        (item_number, text_on_this_line, junk_prefix), or None if this is not an item
        start. The text is normally "" because the number sits alone on its line.
    """
    match = _ITEM_START_RE.match(line)
    if match is None:
        return None

    item_number = int(match["number"])
    text_on_line = match["rest"].strip()
    junk = match["junk"]

    # Item 0 does not exist, and a large number is wrapped prose or OCR noise
    if not 1 <= item_number <= _MAX_ITEM_NUMBER:
        return None

    # An abbreviation's period is not a sentence end, so a number after it is a wrap
    previous_word = _last_word(previous_line)
    if previous_word in _ABBREVIATIONS_BEFORE_NUMBER:
        return None

    # Section banners are exempt from "unfinished": they end on words like "ITEMS" with
    # no punctuation at all
    previous_is_unfinished = not _ends_a_sentence(previous_line) and not _is_section_header(
        previous_line
    )

    # A number that continues an unfinished line is a wrap too
    if previous_is_unfinished and previous_word in _NUMBER_PRECEDING_WORDS:
        return None

    # The number alone on its line is the normal form and needs no further evidence
    if not text_on_line:
        return item_number, text_on_line, junk

    # Text on the same line is where wrapped prose looks convincing, so it also has to
    # start a sentence and follow one
    if not text_on_line[0].isupper():
        return None
    if previous_line and previous_is_unfinished:
        return None

    return item_number, text_on_line, junk


# ---------------------------------------------------------------------------
# SIZING AND METADATA
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token count from word count, rounded up so chunks stay under the cap."""
    return int(len(text.split()) * _TOKENS_PER_WORD) + 1


def _split_long_text(text: str, max_tokens: int = _MAX_CHUNK_TOKENS) -> list[str]:
    """Cut an over-long item at sentence boundaries into parts that fit the embedder.

    Eminent domain items carry pages of legal property descriptions, and the budget item
    spans several pages. A single sentence longer than the budget is kept whole.
    """
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0

    # Fill each part with sentences until the next one would overflow the budget
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence_tokens = _estimate_tokens(sentence)

        if current and current_tokens + sentence_tokens > max_tokens:
            parts.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        parts.append(" ".join(current))

    return parts


def _unique(values: list[str]) -> list[str]:
    """Drop duplicates, keeping first-seen order."""
    return list(dict.fromkeys(values))


def _find_numbers(pattern: re.Pattern[str], text: str) -> list[str]:
    """Every <number> group the pattern captures, with OCR's stray spaces removed."""
    return [
        "".join(match["number"].split())
        for match in pattern.finditer(text)
    ]


def _extract_metadata(text: str) -> dict[str, Any]:
    """Pull identifiers and the outcome out of an item's text.

    Embeddings are poor at exact lookups: a question naming "Ordinance No. 20251120-008"
    or "District 5" is better answered by filtering on these fields than by similarity.
    Lists are empty and scalars None when nothing is found.
    """
    vote = _VOTE_RE.search(text)
    lowered = text.lower()

    # First keyword wins, so _OUTCOME_KEYWORDS is ordered most specific first
    outcome = next(
        (label for keyword, label in _OUTCOME_KEYWORDS if keyword in lowered), None
    )

    return {
        "ordinance_numbers": _unique(_find_numbers(_ORDINANCE_RE, text)),
        "resolution_numbers": _unique(_find_numbers(_RESOLUTION_RE, text)),
        "case_numbers": _unique(_CASE_RE.findall(text)),
        "districts": _unique(_DISTRICT_RE.findall(text)),
        "dollar_amounts": _unique(_DOLLAR_RE.findall(text))[:10],
        "vote": f"{vote['yes']}-{vote['no']}" if vote else None,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------


def split_into_items(lines: list[tuple[str, int]]) -> list[dict[str, Any]]:
    """Group (line, page) pairs into agenda items, tracking the section each sits under.

    Anything before the first numbered item (proclamations, call to order) becomes one
    "meeting info" record so that meeting-level questions still have something to hit.

    Returns:
        One record per item with item_number, section, text_lines, page_start, page_end.
    """
    items: list[dict[str, Any]] = []
    section = "FRONT MATTER"
    current: dict[str, Any] | None = None
    preamble: list[str] = []
    preamble_pages: list[int] = []
    previous_line = ""
    previous_was_header = False
    last_number: int | None = None

    for line, page in lines:
        item_start = _is_item_start(line, previous_line)

        if item_start is not None:
            # A new item closes the previous one
            if current is not None:
                items.append(current)

            item_number, first_text, junk = item_start
            item_number = _repair_item_number(item_number, junk, last_number)
            last_number = item_number
            previous_was_header = False
            current = {
                "item_number": item_number,
                "section": section,
                "text_lines": [first_text] if first_text else [],
                "page_start": page,
                "page_end": page,
            }

        elif _is_section_header(line):
            # Banners change the heading but never belong to an item's text. A long banner
            # wraps onto a second line, e.g. "ZONING ... (PUBLIC HEARINGS AND" /
            # "POSSIBLE ACTION)", so an unfinished banner absorbs the line after it.
            if previous_was_header and _is_unfinished_header(section):
                section = f"{section} {line}"
            else:
                section = line
            previous_was_header = True

            if current is not None:
                items.append(current)
                current = None

        elif current is not None:
            current["text_lines"].append(line)
            current["page_end"] = page
            previous_was_header = False

        else:
            preamble.append(line)
            preamble_pages.append(page)
            previous_was_header = False

        previous_line = line

    if current is not None:
        items.append(current)

    if preamble:
        items.insert(
            0,
            {
                "item_number": None,
                "section": "MEETING INFO",
                "text_lines": preamble,
                "page_start": min(preamble_pages),
                "page_end": max(preamble_pages),
            },
        )

    return items


def chunk_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Convert one minutes PDF into finished chunk records ready to embed.

    Each record has a chunk_id, the text, and flat metadata fields.
    """
    items = split_into_items(_extract_lines(pdf_path))
    meeting_date = _meeting_date_from_name(pdf_path)

    chunks: list[dict[str, Any]] = []
    occurrences: dict[int | None, int] = {}

    for item in items:
        text = _join_lines(item["text_lines"])

        # An item marker with no text under it (OCR noise) has nothing to retrieve
        if not text.strip():
            continue

        # An item can appear twice (pulled from consent, discussed later), so each
        # occurrence is numbered rather than overwritten
        occurrence = occurrences.get(item["item_number"], 0) + 1
        occurrences[item["item_number"]] = occurrence

        label = f"Item {item['item_number']}" if item["item_number"] is not None else "Meeting info"
        parts = _split_long_text(text)

        for part_number, part_text in enumerate(parts, start=1):
            chunk_id = (
                f"{meeting_date or pdf_path.stem}:{label.replace(' ', '')}"
                f":occ{occurrence}:part{part_number}"
            )
            chunk = {
                "chunk_id": chunk_id,
                "text": part_text,
                "source_file": pdf_path.name,
                "meeting_date": meeting_date,
                "meeting_type": "Regular Council Meeting",
                "section": item["section"],
                "item_number": item["item_number"],
                "item_occurrence": occurrence,
                "chunk_part": part_number,
                "chunk_parts_total": len(parts),
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "estimated_tokens": _estimate_tokens(part_text),
            }
            chunk.update(_extract_metadata(part_text))
            chunks.append(chunk)

    return chunks


def context_header(chunk: dict[str, Any]) -> str:
    """One-line provenance header embedded with the chunk text.

    Without it, a chunk reading "approved on consent on an 11-0 vote" carries no hint of
    which meeting or item it belongs to, and matches every other vote in the corpus.
    Example: "Austin City Council, Regular Council Meeting, 2025-11-20 | Consent Agenda | Item 22"
    """
    pieces = [
        f"Austin City Council, {chunk['meeting_type']}, {chunk['meeting_date']}",
        chunk["section"].title(),
    ]

    if chunk["item_number"] is not None:
        pieces.append(f"Item {chunk['item_number']}")
    if chunk["chunk_parts_total"] > 1:
        pieces.append(f"part {chunk['chunk_part']} of {chunk['chunk_parts_total']}")

    return " | ".join(pieces)


def embedding_text(chunk: dict[str, Any]) -> str:
    """Exactly the string that gets embedded for a chunk: header, newline, text."""
    return f"{context_header(chunk)}\n{chunk['text']}"


def build_chunks(docs_dir: Path, out_file: Path) -> list[dict[str, Any]]:
    """Chunk every PDF in a folder and write the result as JSON Lines.

    JSON Lines keeps line N of the file aligned with row N of the embedding matrix, which
    is what lets search return metadata alongside each vector hit.

    Raises:
        FileNotFoundError: If the folder has no PDFs in it.
    """
    pdf_paths = sorted(path for path in docs_dir.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}")

    all_chunks: list[dict[str, Any]] = []

    # One PDF at a time, so a malformed file is easy to spot in the printed summary
    for pdf_path in pdf_paths:
        chunks = chunk_pdf(pdf_path)
        all_chunks.extend(chunks)
        print(f"{pdf_path.name}: {len(chunks)} chunks")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks to {out_file}")
    return all_chunks


def load_chunks(chunks_file: Path = DEFAULT_CHUNKS_FILE) -> list[dict[str, Any]]:
    """Read chunks back in file order, which matches embedding row order.

    Raises:
        FileNotFoundError: If the file does not exist yet.
    """
    if not chunks_file.exists():
        raise FileNotFoundError(f"{chunks_file} not found. Run: python prep.py")

    with chunks_file.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    """Chunk the PDFs from the command line."""
    parser = argparse.ArgumentParser(
        description="Chunk City of Austin council minutes PDFs by agenda item."
    )
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_DIR, help="Folder of PDFs.")
    parser.add_argument("--out", type=Path, default=DEFAULT_CHUNKS_FILE, help="Output .jsonl.")
    args = parser.parse_args()

    build_chunks(args.docs, args.out)


if __name__ == "__main__":
    main()
