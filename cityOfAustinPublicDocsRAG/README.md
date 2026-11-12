# City of Austin Public Docs RAG

Semantic search over Austin City Council meeting minutes. The minutes are split by
agenda item, each item is embedded locally, and a question retrieves the closest items
with the metadata needed to cite them.

This is the retrieval half of a RAG system. Nothing here calls an LLM yet.

## Setup

```powershell
python setup_env.py              # creates .venv and installs requirements.txt
.venv\Scripts\Activate.ps1       # activate it
```

On the first embed the model (~130 MB) downloads from Hugging Face and is cached in
`~/.cache/huggingface`. Everything after that runs offline.

## Build the index

```powershell
python prep.py                   # coaDocs/*.PDF  -> index/chunks.jsonl
python embed.py --build          # chunks.jsonl   -> index/embeddings.npy
```

Re-run **both** after adding a PDF. `search.py` refuses to run if the two files disagree
on chunk count, rather than returning the wrong metadata for a hit.

## Ask your own questions

```powershell
python search.py "How much was the Elena Torres settlement?"
python search.py "zoning changes on Dalton Lane" --k 3
python search.py "eminent domain near Barton Creek" --full      # whole chunks
python search.py "kennel rules" --json                          # all metadata as JSON
```

| Flag | Default | Meaning |
|---|---|---|
| `--k N` | 5 | How many chunks to return |
| `--full` | off | Print whole chunks instead of 400-character previews |
| `--json` | off | Raw JSON including every metadata field |
| `--chunks`, `--vectors` | `index/...` | Point at a different index |

## Other commands

```powershell
python embed.py --stats          # real token counts per chunk
python test_retrieval.py         # 15 questions with hand-verified answers
python test_retrieval.py --k 1 --verbose
python prep.py --docs otherDocs --out index/other.jsonl
```

Every CLI uses `argparse` from the standard library. No third-party CLI packages.

## Files

| File | Role |
|---|---|
| `prep.py` | Extract PDF text, clean it, split by agenda item, attach metadata |
| `embed.py` | Embed chunks and queries; build and load the numpy index |
| `search.py` | Cosine similarity against the index; returns top k with metadata |
| `test_retrieval.py` | Known questions with known answers, to measure retrieval |
| `setup_env.py` | Create `.venv` and install `requirements.txt` |
| `index/` | Generated: `chunks.jsonl` and `embeddings.npy` (gitignored) |

## How it works

```
PDF -> lines tagged with page numbers -> split on item numbers -> metadata
    -> chunks.jsonl ──┬─> embed with a context header -> embeddings.npy
                      │
query -> embed ───────┴─> dot product over all rows -> top 5 chunks + metadata
```

**Where metadata lives and how it comes back.** Each chunk is one JSON object on one line
of `index/chunks.jsonl`, holding its text *and* its metadata together. `embeddings.npy` is
a `(n_chunks, 384)` float32 matrix whose **row N corresponds to line N** of the JSONL file.
That shared index is the only link between a vector and its metadata, so nothing may
reorder one without the other. A search embeds the question, takes the dot product against
the whole matrix, picks the best row indices, and looks those same indices up in the
in-memory chunk list. The returned dict is the chunk record itself plus a `score` field,
so retrieval hands back text and metadata in one object with no second lookup and no
database. Fields per chunk:

`chunk_id`, `text`, `source_file`, `meeting_date`, `meeting_type`, `section`,
`item_number`, `item_occurrence`, `chunk_part`, `chunk_parts_total`, `page_start`,
`page_end`, `estimated_tokens`, `ordinance_numbers`, `resolution_numbers`,
`case_numbers`, `districts`, `dollar_amounts`, `vote`, `outcome`.

**Is 512 tokens enough?** Yes, with room to spare. Measured on the two PDFs
(`python embed.py --stats`):

| | tokens |
|---|---|
| Median chunk | 163 |
| Mean chunk | 190 |
| 95th percentile | 359 |
| Largest chunk | 409 |
| Chunks truncated at 512 | 0 |
| Whole corpus (143 chunks) | ~27,200 |

An item is naturally one or two paragraphs, which is well inside the limit. Only three
items were long enough to need splitting: the amended budget item and two eminent-domain
items with pages of property descriptions. For the generation step you are planning,
top-5 retrieval adds roughly **1,800 tokens** of context, and even top-20 would only be
about 7,000 — small next to a 200K context window. The whole corpus is ~27K tokens, so
two meetings would fit in a prompt outright; RAG starts to earn its place at roughly 20+
meetings, and the retrieval quality matters more than the token budget.

**Why each chunk carries a context header.** The string that gets embedded is prefixed
with its provenance:

```
Austin City Council, Regular Council Meeting, 2025-11-20 | Consent Agenda | Item 22
Authorize a contract for construction services for the Cargo Avenue ...
```

Without it, a chunk reading "approved on consent on an 11-0 vote" is indistinguishable
from every other vote in the corpus.

## What the documents forced

The two PDFs are structurally different: the 2025 file has a real text layer, the 2026
file is a scan with OCR errors. The splitter handles both:

- Item numbers sit alone on their own line, with text following.
- Running headers, page numbers, and the stray date line are stripped before splitting,
  because items routinely straddle a page break.
- Items appear out of order. Items 4, 5, and 6 were pulled from the 2025 consent agenda
  and show up 20 pages later, and item 4 appears **twice**, so each occurrence is kept
  separately and numbered via `item_occurrence`.
- Wrapped text that looks like an item start is rejected: a bare `2.` left by
  "District(s) Affected: District 2." and a `2,` left by "Abstract No. 2, Travis County".
- OCR noise is tolerated: `8,` for `8.`, and `] 8.` where the OCR read item 18's leading
  `1` as a bracket. The number repair only fires when the result is exactly the next item
  in sequence.
- Soft hyphens are rejoined, so "no- cost" and "Harper- Madison" don't reach the index.

Verify a parse with:

```powershell
python -c "import json; rows=[json.loads(l) for l in open('index/chunks.jsonl',encoding='utf-8')]; print(sorted({(r['meeting_date'],r['item_number']) for r in rows}))"
```

Current state: the 2025 file yields items 1–76 with no gaps and no false item starts
(items 4, 5, and 6 appearing in their later discussion positions, and item 4 twice). The
2026 file yields items 1–53 except item 11, which is absent from the document itself
(it goes 10 → 12).

## Known limits, and what to add next

1. **Exact identifiers are the weak spot.** Ask for "Pilot Knob Municipal Utility District
   No. 5" and three near-identical items score within 0.003 of each other. Pure semantic
   search cannot reliably separate them. The fix is **hybrid search**: add BM25 keyword
   scoring over the same chunks and combine the two rankings. This is the first thing to
   build next.
2. **Metadata is collected but never used to filter.** Adding `--district`, `--after`, or
   `--section` filters to `search.py` is cheap and would sharpen narrow questions.
3. **No generation step yet.** The natural next piece: pass the top k chunks to Claude with
   an instruction to answer only from them and cite `meeting_date` plus `item_number`.
4. **Flat numpy scan.** Fine to roughly 100K chunks. A vector database earns its place
   only past that.
5. **Scanned-PDF text quality.** The 2026 OCR mangles some names ("Velisquez", "Josd").
   Retrieval copes, but quoted text from that file should be checked against the PDF.
