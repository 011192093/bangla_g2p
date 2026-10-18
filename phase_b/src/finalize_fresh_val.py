"""
Final sanity pass before freezing the fresh-val set: confirms the merged
file is exactly 600 rows, 3 columns, no malformed rows, zero decomposed
y়/ড়/ঢ় left anywhere, no duplicate words, then writes it to --out and
prints the SHA256 so it can be recorded and the file frozen.

Usage:
    python finalize_fresh_val.py --reviewed fresh_val_merged.tsv \
        --out ../data/lexicon_fresh_val.tsv
"""
import argparse
import hashlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from normalizer import compose_nukta

NUKTA_DECOMPOSED = ("য়", "ড়", "ঢ়")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected_n", type=int, default=600)
    args = ap.parse_args()

    out_path = args.out if args.out.endswith(".tsv") else args.out + ".tsv"

    with io.open(args.reviewed, encoding="utf-8") as f:
        lines = f.readlines()

    rows = []
    malformed = []
    for i, line in enumerate(lines, 1):
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            malformed.append((i, line))
            continue
        word, phon, tag = parts
        word = compose_nukta(word)
        rows.append((word, phon, tag))

    print(f"Loaded {len(rows)} well-formed rows ({len(malformed)} malformed)")
    for i, line in malformed:
        print(f"  MALFORMED line {i}: {line!r}")

    still_decomposed = [w for w, _, _ in rows if any(s in w for s in NUKTA_DECOMPOSED)]
    seen = {}
    dupes = []
    for w, p, t in rows:
        if w in seen:
            dupes.append(w)
        seen[w] = (p, t)

    ok = True
    if len(rows) != args.expected_n:
        print(f"FAIL: expected {args.expected_n} rows, got {len(rows)}")
        ok = False
    if malformed:
        print(f"FAIL: {len(malformed)} malformed row(s)")
        ok = False
    if still_decomposed:
        print(f"FAIL: {len(still_decomposed)} word(s) still have decomposed nukta: {still_decomposed[:5]}")
        ok = False
    if dupes:
        print(f"FAIL: {len(dupes)} duplicate word(s): {dupes[:5]}")
        ok = False

    if not ok:
        print("\nNot writing output -- fix the issues above first.")
        sys.exit(1)

    with io.open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for word, phon, tag in rows:
            f.write(f"{word}\t{phon}\t{tag}\n")

    sha256 = hashlib.sha256()
    with io.open(out_path, "rb") as f:
        sha256.update(f.read())

    print(f"\nAll checks passed. Wrote {len(rows)} rows to {out_path}")
    print(f"SHA256: {sha256.hexdigest()}")
    print("Freeze this file -- any future edit must go through a new reviewed")
    print("draft + merge, not a hand-edit of lexicon_fresh_val.tsv directly.")


if __name__ == "__main__":
    main()
