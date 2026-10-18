#!/usr/bin/env python
"""
Classify why two models disagree, using the discordant file from compare_dumps.py --out.

For each word, the LOSING model's prediction is compared with gold token by token:
  vowel_quality : only o<->O, e<->E, a<->A, A<->E substitutions
  schwa         : only an inserted/deleted inherent vowel (o or O)
  vowel+schwa   : both of the above, nothing else
  s/sh          : only s<->sh substitutions (possibly plus vowel/schwa)
  other         : anything else (consonant errors, garbled output, typos in input)

The first three (plus s/sh) are convention-type disagreements: plausible readings that
differ from the label. "other" is the closest thing to a real error.

  python classify_discordant.py val_discordant.tsv
  python classify_discordant.py fresh_discordant.tsv --show other
"""
import argparse
import csv
import difflib
from collections import Counter

QUAL = {frozenset("oO"), frozenset("eE"), frozenset("aA"), frozenset("AE")}
SCHWA = {"o", "O"}
SIB = {frozenset(["s", "sh"])}


def classify(gold, pred):
    g, p = gold.split(), pred.split()
    kinds = set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, g, p, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        gs, ps = g[i1:i2], p[j1:j2]
        if op == "replace" and len(gs) == len(ps):
            for a, b in zip(gs, ps):
                pair = frozenset([a, b])
                if pair in QUAL:
                    kinds.add("vq")
                elif pair in SIB:
                    kinds.add("sib")
                else:
                    kinds.add("other")
        elif op in ("insert", "delete") and all(t in SCHWA for t in gs + ps):
            kinds.add("schwa")
        else:
            kinds.add("other")
    if "other" in kinds:
        return "other"
    if "sib" in kinds:
        return "s/sh"
    if kinds == {"vq"}:
        return "vowel_quality"
    if kinds == {"schwa"}:
        return "schwa"
    return "vowel+schwa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--show", help="print the words of one category")
    a = ap.parse_args()

    with open(a.file, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    pred_cols = [c for c in rows[0] if c.endswith("_pred")]
    names = [c[:-5] for c in pred_cols]

    counts = {n: Counter() for n in names}
    shown = []
    for r in rows:
        winner = r["who_right"]
        loser = [n for n in names if n != winner][0]
        cat = classify(r["gold"], r[f"{loser}_pred"])
        counts[winner][cat] += 1
        if a.show and cat == a.show:
            shown.append((r["word"], r["gold"], loser, r[f"{loser}_pred"]))

    cats = ["vowel_quality", "schwa", "vowel+schwa", "s/sh", "other"]
    print(f"{'why the loser was wrong':<24}" + "".join(f"{n + ' won':>16}" for n in names))
    for c in cats:
        print(f"{c:<24}" + "".join(f"{counts[n][c]:16d}" for n in names))
    print(f"{'total':<24}" + "".join(f"{sum(counts[n].values()):16d}" for n in names))
    conv = {n: sum(counts[n][c] for c in cats[:4]) for n in names}
    print(f"{'convention-type (1-4)':<24}" + "".join(f"{conv[n]:16d}" for n in names))

    if shown:
        print(f"\n{a.show} cases:")
        for w, g, l, p in shown:
            print(f"  {w}\tgold: {g}\t{l}: {p}")


if __name__ == "__main__":
    main()
