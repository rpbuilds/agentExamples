---
name: python-style
description: Use whenever writing, editing, or reviewing Python (.py) code in this repo, including new files, new functions, small fixes, and refactors. Optimizes for a reader skimming the file for the first time: annotations only where they inform, docstrings that say why, every regex shown with an example of what it matches, and lookup tables instead of if-chains. Do not use for non-Python files.
---

# Python style for this repo

These files are read far more often than they are run. Every rule below serves one
reader: someone opening the file for the first time, trying to answer "what does this
do, and can I trust it?"

The test for any line is **does a first-time reader learn something from it?** If not,
it is noise, even when it is technically correct. Apply these rules to any Python you
write or change. When you edit an existing file, bring the code you touch up to these
rules; leave untouched code alone unless asked.

## 1. Type annotations where they inform

Annotate **every function signature**: all parameters, and the return type including
`-> None`. Annotate class attributes and instance attributes in `__init__`. That is the
contract, and a reader should never have to read a body to learn it.

Inside a function body, annotate a local **only when the type is not obvious from the
line itself**:

```python
# Annotate: an empty container's element type is invisible
chunks: list[dict[str, Any]] = []
seen: dict[int | None, int] = {}

# Annotate: the value crosses a boundary that returns Any
payload: dict[str, Any] = json.loads(raw)

# Do NOT annotate: the right-hand side already says it
match = _ITEM_START_RE.match(line)
words = line.split()
total = sum(counts)
item_number = int(match.group("number"))
```

**Never pre-declare a loop target or an unpacking target just to carry a type.** Lines
like `line: str` above `for line in lines:`, or `pattern: re.Pattern[str]` above a
`for pattern, replacement in ...`, cost a line each and tell the reader nothing the
iterable did not already say. If an element type is genuinely unclear, annotate the
*iterable* where it was built, or pick a better loop-variable name.

Use built-in generics (`list[str]`, `dict[str, Any]`, `X | None`). Never `List`,
`Dict`, or `Optional`. Use `Any` only at real boundaries (raw JSON, third-party
responses) and narrow it immediately.

## 2. Docstrings that carry their weight

- **Every module** opens with a docstring: what the file is for, and how to run it if
  it is runnable.
- **Every class and public function** has a docstring whose first line says *why it
  exists*, not what the code does. If that line could be derived from the function
  name, rewrite it or drop it.
- Add `Args:` / `Returns:` / `Raises:` only for what the signature cannot say: units,
  formats, ranges, invariants, the shape inside a `dict[str, Any]`, which exceptions
  are raised on purpose. Never restate a parameter's name as its description.
- **Private helpers get one line.** A docstring must never be longer than the function
  it documents.

```python
# Good: one line, says why
def _ends_a_sentence(line: str) -> bool:
    """True when a line finishes a sentence, allowing for a closing quote mark."""


# Bad: six lines of ceremony for a one-line body
def _last_word(line: str) -> str:
    """Return the final word of a line, lowercased and stripped of punctuation.

    Args:
        line: The line.

    Returns:
        The last word.
    """
```

## 3. Every regex comes with an example

A regex is the least readable thing in a Python file. Three rules, no exceptions.

**a. Show what it matches.** Directly above every pattern, give real sample strings
from the actual data and what each should produce. One line per interesting case. This
matters more than any prose description of the pattern.

**b. Use `re.VERBOSE` whenever a pattern has more than two constructs**, one logical
piece per line, each with a comment:

```python
# Start of an agenda item. The number normally sits alone on its own line.
#   "22."              -> number 22, rest ""
#   "] 8."             -> junk "]", number 8, rest ""   (scan ate the leading 1)
#   "7, Approve a..."  -> number 7, rest "Approve a..."
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
```

**c. Name every capture group** with `(?P<name>...)` and read it back by name. Never
index groups by number in code a reader has to follow.

Keep patterns small. If one needs more than about six VERBOSE lines, split it or do
part of the work in plain Python — a short function with an `if` reads better than a
pattern with lookarounds. Never feed one regex's output into another without naming the
intermediate value.

## 4. Lookup tables instead of if-chains

When several branches do the same thing to different fields or values, write the cases
as data and loop over them. The reader then sees the shape at a glance.

```python
# Good: the cases are data
_IDENTIFIER_LABELS = (
    ("ordinance_numbers", "Ordinance"),
    ("case_numbers", "Case"),
    ("districts", "District"),
)

parts = [f"{label} {', '.join(hit[key])}" for key, label in _IDENTIFIER_LABELS if hit[key]]

# Bad: six near-identical if blocks the reader has to diff by eye
if hit["ordinance_numbers"]:
    parts.append(f"Ordinance {', '.join(hit['ordinance_numbers'])}")
if hit["case_numbers"]:
    ...
```

## 5. Naming (PEP 8)

| Kind | Style | Example |
|---|---|---|
| Constants (module-level, never reassigned) | `UPPER_SNAKE_CASE` | `MAX_STEPS: int = 8` |
| Variables, functions, methods | `snake_case` | `fetch_weather()` |
| Classes, type aliases | `PascalCase` | `WeatherClient` |
| Internal-only (module, function, attribute) | leading `_` | `_parse_response()` |

- Prefix anything callers outside the module or class should not use with `_`.
- No `__double_leading` names unless you need name mangling.
- Spell names out: `temperature`, not `tmp`. Single letters only in short
  comprehensions and indices.
- A name that makes a comment unnecessary beats a short name plus a comment.

## 6. Imports

Three groups, in this order, one blank line between, alphabetical inside each:

1. Standard library
2. Third-party packages
3. Local/first-party modules from this repo

Plain `import x` before `from x import y` within a group. No wildcard imports. Remove
unused imports.

## 7. Comments

- Comment the **why**, never the what. A comment that paraphrases the next line is
  worse than none: it doubles the reading and it goes stale.
- Comment a block when it encodes a decision, a workaround, or a fact about the data
  the code cannot show — why this threshold, which bug this guards, what the source
  file does wrong. Those are the most valuable lines in the file. Keep them.
- Explain magic numbers inline or make them named constants.
- Do not banner-comment short files. In a file over ~200 lines use a handful at most:
  ```python
  # ---------------------------------------------------------------------------
  # TOOL DEFINITIONS
  # ---------------------------------------------------------------------------
  ```
- If a function needs a comment every few lines to be followable, split the function.

## 8. General conventions

- 4-space indent. Lines at 100 characters or fewer.
- f-strings for formatting; `pathlib.Path` instead of `os.path`.
- Catch specific exceptions. Never a bare `except:`.
- Guard entry points with `if __name__ == "__main__":` and a `main() -> None`.
- Module layout: docstring, imports, constants, private helpers, public API, entry point.
- Prefer a flat sequence of named steps over nesting. Return early instead of wrapping a
  whole function body in an `if`.

## Checklist before finishing

- [ ] Every signature is fully annotated; no local annotation that merely repeats its
      right-hand side, and no pre-declared loop targets
- [ ] Every module, class, and public function says *why* it exists; private helpers
      are one line
- [ ] Every regex has a sample-input comment, named groups, and `re.VERBOSE` if it is
      more than two constructs
- [ ] Repeated branches collapsed into a table or a loop
- [ ] Imports grouped and sorted; names spelled out in full
- [ ] Every comment still in the file tells the reader something the code does not
