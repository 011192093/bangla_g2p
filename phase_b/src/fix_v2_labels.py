#!/usr/bin/env python
"""
Fix three symbol conventions in a WORKING (v2) label file. Never run on the frozen files.

  1. ঐ / ৈ  : "o i" -> "O i"   (closed o)
  2. ঔ / ৌ  : "o u" -> "O u"
  3. N      : -> n             (ণ and ষ্ণ are n)
  4. optional --merge_ae: A -> E  (both are /ae/; one symbol for one sound)

Labels are written in canonical form (atomic phonemes). Rules 1-2 are applied only when the
number of "o i" (or "o u") in the label equals the number of ঐ/ৈ (or ঔ/ৌ) in the spelling;
otherwise the row is left unchanged and listed for manual review.

  python fix_v2_labels.py --file ../data/lexicon_split_train.tsv --out ../data/lexicon_split_train_v2b.tsv --merge_ae
"""
import argparse
import csv
import sys
from collections import Counter

from canon_rescore import canon

OI_CHARS = ("ঐ", "ৈ")   # ঐ, ৈ
OU_CHARS = ("ঔ", "ৌ")   # ঔ, ৌ


def count_pair(toks, a, b):
    return sum(1 for i in range(len(toks) - 1) if toks[i] == a and toks[i + 1] == b)


def replace_pair(toks, a, b, new_a):
    out, i = [], 0
    while i < len(toks):
        if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
            out += [new_a, b]
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge_ae", action="store_true")
    ap.add_argument("--review", default="fix_v2_review.tsv")
    a = ap.parse_args()
    if "frozen" in a.file.replace("\\", "/"):
        sys.exit("refusing to touch a frozen file")

    stats, review, lines = Counter(), [], []
    with open(a.file, encoding="utf-8-sig") as f:
        raw = f.read().split("\n")
    for line in raw:
        p = line.rstrip("\r").split("\t")
        if len(p) < 2 or not p[1].strip() or p[0].strip().lower().lstrip("#").strip() == "word":
            lines.append(line.rstrip("\r"))
            continue
        word, toks = p[0], canon(p[1]).split()
        before = list(toks)
        for chars, vowel, name in ((OI_CHARS, "i", "oi"), (OU_CHARS, "u", "ou")):
            n_sp = sum(word.count(c) for c in chars)
            if not n_sp:
                continue
            n_open = count_pair(toks, "o", vowel)
            if n_open == 0:
                continue
            if n_open == n_sp:
                toks = replace_pair(toks, "o", vowel, "O")
                stats[name] += 1
            else:
                review.append((word, p[1], f"{name}: {n_sp} in spelling, {n_open} 'o {vowel}' in label"))
                stats[name + "_review"] += 1
        if "N" in toks:
            toks = ["n" if t == "N" else t for t in toks]
            stats["N"] += 1
        if a.merge_ae and "A" in toks:
            toks = ["E" if t == "A" else t for t in toks]
            stats["A->E"] += 1
        if toks != before:
            stats["rows_changed"] += 1
        p[1] = " ".join(toks)
        lines.append("\t".join(p))

    with open(a.out, "x", encoding="utf-8", newline="\n") as g:
        g.write("\n".join(lines))
    with open(a.review, "w", encoding="utf-8", newline="") as g:
        w = csv.writer(g, delimiter="\t")
        w.writerow(["word", "label", "reason"])
        w.writerows(review)
    print(f"written {a.out}")
    for k in ("oi", "ou", "N", "A->E", "oi_review", "ou_review", "rows_changed"):
        print(f"  {k:<13} {stats[k]}")
    print(f"rows needing manual review: {len(review)} -> {a.review}")


if __name__ == "__main__":
    main()
