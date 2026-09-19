"""Resolve a conflict in ASTRA_DISCUSSION.md by merging entries, never by dropping them.

Both sides append new `### ` entries at the top of the log, so a line-based merge conflicts every time.
This splits each side into whole entries, keeps the union, deduplicates by heading and orders newest
first by the timestamp in the heading. Nothing is discarded: an entry present on either side survives.
"""

import re
import sys
from pathlib import Path

HEADING = re.compile(r"^### (\d{4}-\d{2}-\d{2})(?: (\d{2}:\d{2}) UTC)?, ", re.M)


def split_entries(body):
    starts = [match.start() for match in HEADING.finditer(body)]
    if not starts:
        return [], body
    preamble = body[: starts[0]]
    bounds = starts + [len(body)]
    return [body[bounds[i] : bounds[i + 1]].rstrip() + "\n" for i in range(len(starts))], preamble


def sort_key(entry):
    match = HEADING.match(entry)
    return (match.group(1), match.group(2) or "00:00")


def merge(text):
    if "<<<<<<<" not in text:
        return text
    start = text.index("<<<<<<< ")
    head_start = text.index("\n", start) + 1
    mid = text.index("\n=======\n", head_start)
    end = text.index("\n>>>>>>>", mid)
    tail = text.index("\n", end + 1) + 1
    ours, _ = split_entries(text[head_start : mid + 1])
    theirs, _ = split_entries(text[mid + len("\n=======\n") : end + 1])
    seen, merged = set(), []
    for entry in ours + theirs:
        heading = entry.splitlines()[0]
        if heading in seen:
            continue
        seen.add(heading)
        merged.append(entry)
    merged.sort(key=sort_key, reverse=True)
    return text[:start] + "\n".join(merged) + "\n" + text[tail:]


def dedupe_whole_file(text):
    """A heading may appear twice if one copy sat outside the conflict region. Keep the first, drop repeats."""
    entries, preamble = split_entries(text)
    if not entries:
        return text
    seen, kept = set(), []
    for entry in entries:
        heading = entry.splitlines()[0]
        if heading in seen:
            continue
        seen.add(heading)
        kept.append(entry)
    kept.sort(key=sort_key, reverse=True)
    return preamble + "\n".join(kept)


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/iclr/handoff/ASTRA_DISCUSSION.md")
    text = path.read_text()
    while "<<<<<<<" in text:
        text = merge(text)
    text = dedupe_whole_file(text)
    path.write_text(text)
    headings = [line for line in text.splitlines() if line.startswith("### ")]
    print(f"merged, {len(headings)} entries preserved:")
    for heading in headings:
        print("  ", heading[4:100])


if __name__ == "__main__":
    main()
