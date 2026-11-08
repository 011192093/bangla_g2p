#!/usr/bin/env python
"""
Random spot-check of label changes between two versions of a split.

  python spot_check.py --old ../data/frozen_2026-10-25/lexicon_split_train.tsv \
      --new ../data/lexicon_split_train.tsv --n 50 --out spot_check_train.csv

Prints the sample and writes a CSV (opens in Excel) with columns:
  id, word, register, old_label, new_label, change_ok
Fill change_ok with y (the new label is right) or n (the change is wrong).
"""
import argparse
import csv
import random
import unicodedata


def norm(w):
    w = unicodedata.normalize("NFC", w.strip())
    for a, b in {"য়": "য়", "ড়": "ড়", "ঢ়": "ঢ়"}.items():
        w = w.replace(a, b)
    return w


def read(path):
    d = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 2 or p[0].strip().lower().lstrip("#").strip() == "word":
                continue
            d[norm(p[0])] = (" ".join(p[1].split()), p[2].strip() if len(p) > 2 else "")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="spot_check.csv")
    a = ap.parse_args()

    old, new = read(a.old), read(a.new)
    common = [w for w in new if w in old]
    label_changed = [w for w in common if new[w][0] != old[w][0]]
    reg_changed = [w for w in common if new[w][1] != old[w][1]]
    print(f"words in both: {len(common)}   label changed: {len(label_changed)}   "
          f"register changed: {len(reg_changed)}   "
          f"only in old: {len(set(old) - set(new))}   only in new: {len(set(new) - set(old))}")

    rng = random.Random(a.seed)
    sample = rng.sample(label_changed, min(a.n, len(label_changed)))
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "word", "register", "old_label", "new_label", "change_ok"])
        for i, w in enumerate(sample, 1):
            wr.writerow([i, w, new[w][1], old[w][0], new[w][0], ""])
            print(f"{i:3d}  {w}\n     old: {old[w][0]}\n     new: {new[w][0]}")
    print(f"\nsheet written: {a.out}  (fill change_ok with y / n)")


if __name__ == "__main__":
    main()
