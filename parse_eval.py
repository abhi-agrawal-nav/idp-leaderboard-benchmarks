#!/usr/bin/env python3
"""Turn evaluate.py stdout into one markdown table row.

  python benchmarks/olmocr/evaluate.py --model qwen35-2b | python parse_eval.py --name qwen35-2b
  python parse_eval.py --header        # prints the table header + separator

Pure stdlib. The 8 category pass-rates are read from the "Results by category" block only
(the "Results by test type" block carries a near-identical 'baseline' line we must ignore),
and the overall from the "OVERALL SCORE" line.
"""

import argparse
import re
import sys

CATEGORIES = [
    "arxiv_math", "old_scans_math", "headers_footers", "long_tiny_text",
    "old_scans", "multi_column", "tables", "baseline",
]
COLUMNS = ["model", *CATEGORIES, "overall"]

_CAT_RE = re.compile(r"^\s*([a-z_]+)\s*:\s*([\d.]+)%")
_OVERALL_RE = re.compile(r"OVERALL SCORE:\s*([\d.]+)%")


def parse(text):
    """Return {category: 'NN.N', ..., 'overall': 'NN.N'} from evaluate.py output."""
    values = {}
    in_category_block = False
    for line in text.splitlines():
        if "Results by category" in line:
            in_category_block = True
            continue
        if "Results by test type" in line:
            in_category_block = False
            continue
        if in_category_block:
            m = _CAT_RE.match(line)
            if m and m.group(1) in CATEGORIES:
                values[m.group(1)] = m.group(2)
        m = _OVERALL_RE.search(line)
        if m:
            values["overall"] = m.group(1)
    return values


def main():
    ap = argparse.ArgumentParser(description="evaluate.py stdout -> markdown table row")
    ap.add_argument("--name", help="Model name for the row's first cell")
    ap.add_argument("--header", action="store_true", help="Print the table header + separator")
    args = ap.parse_args()

    if args.header:
        print("| " + " | ".join(COLUMNS) + " |")
        print("|" + "|".join(["---"] * len(COLUMNS)) + "|")
        return

    values = parse(sys.stdin.read())
    cells = [args.name or "?"] + [values.get(c, "") for c in CATEGORIES] + [values.get("overall", "")]
    print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
