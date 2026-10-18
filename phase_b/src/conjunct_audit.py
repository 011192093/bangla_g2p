#!/usr/bin/env python
"""
Exact count of merged vs split labels for each conjunct, checked against the SPELLING.

Only words whose spelling actually contains the conjunct (C1 + hasant + C2) are counted,
so ordinary adjacency of two consonants across syllables (no hasant) is excluded.

For each such word the label is classed as:
  merged : contains the merged token        (e.g. "kr")
  split  : contains the two tokens adjacent (e.g. "k r") and not the merged one
  other  : neither (part of a 3-consonant cluster like "ntr", "ndr", a different sound,
           or a label error) - listed with --show_other

  python conjunct_audit.py --file ../data/lexicon_split_train.tsv
  python conjunct_audit.py --file ../data/lexicon_split_train.tsv --show_other ক্র --out split_rows.tsv
"""
import argparse
import csv
import unicodedata
from collections import defaultdict

H = "্"  # hasant
# conjunct -> (merged token, split tokens). Edit to match phoneme_inventory.py.
RULES = {
    "ক" + H + "র": ("kr", ["k", "r"]),
    "প" + H + "র": ("pr", ["p", "r"]),
    "ব" + H + "র": ("br", ["b", "r"]),
    "ত" + H + "র": ("tr", ["t", "r"]),
    "গ" + H + "র": ("gr", ["g", "r"]),
    "ন" + H + "ত": ("nt", ["n", "t"]),
    "স" + H + "ত": ("st", ["s", "t"]),
    "ন" + H + "দ": ("nd", ["n", "d"]),
    "ম" + H + "প": ("mp", ["m", "p"]),
}
NUKTA_FIX = {"য়": "য়", "ড়": "ড়", "ঢ়": "ঢ়"}


def norm(w):
    w = unicodedata.normalize("NFC", w.strip())
    for a, b in NUKTA_FIX.items():
        w = w.replace(a, b)
    return w


def has_pair(toks, pair):
    return any(toks[i:i + len(pair)] == pair for i in range(len(toks) - len(pair) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--show_other", help="conjunct whose 'other' rows to print")
    ap.add_argument("--out", help="TSV of all split rows (conjunct, word, label)")
    a = ap.parse_args()

    counts = defaultdict(lambda: {"merged": 0, "split": 0, "other": 0})
    split_rows, other_rows, words_any_split = [], [], set()
    with open(a.file, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2 or parts[0].strip().lower().lstrip("#").strip() == "word":
                continue
            w, toks = norm(parts[0]), parts[1].split()
            for conj, (merged, split) in RULES.items():
                if conj not in w:
                    continue
                if merged in toks:
                    kind = "merged"
                elif has_pair(toks, split):
                    kind = "split"
                    split_rows.append((conj, w, parts[1]))
                    words_any_split.add(w)
                else:
                    kind = "other"
                    if conj == a.show_other:
                        other_rows.append((w, parts[1]))
                counts[conj][kind] += 1

    print(f"{'conjunct':<10}{'merged':>8}{'split':>8}{'other':>8}{'% split':>9}")
    for conj, (merged, _) in RULES.items():
        c = counts[conj]
        tot = c["merged"] + c["split"]
        pct = c["split"] / tot * 100 if tot else 0
        print(f"{conj:<6}{merged:>4}{c['merged']:8d}{c['split']:8d}{c['other']:8d}{pct:8.1f}%")
    print(f"\nwords with at least one split conjunct: {len(words_any_split)}")

    if other_rows:
        print(f"\n'other' rows for {a.show_other} (first 40):")
        for w, p in other_rows[:40]:
            print(f"  {w}\t{p}")
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="") as g:
            wr = csv.writer(g, delimiter="\t")
            wr.writerow(["conjunct", "word", "label"])
            wr.writerows(split_rows)
        print(f"\nsplit rows written: {a.out}")


if __name__ == "__main__":
    main()
