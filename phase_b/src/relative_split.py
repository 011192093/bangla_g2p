#!/usr/bin/env python
"""
Word-family check: does a model's advantage come from words whose relatives are in train?

For each eval word we find the train word sharing the longest beginning (prefix) and the
train word sharing the longest ending (suffix), after the same normalization as
leak_split.py. overlap = max(prefix, suffix) / length of the eval word.
  e.g. val করেছে vs train করে       -> prefix 3 of 5 letters = 0.60
       val অপরিস্ফুট vs train পরিস্ফুট -> suffix 8 of 9      = 0.89
  1.0 means the whole eval word appears inside a train word (twin, or e.g. val করে / train করেছে).

Primary rule (fixed before looking at results):
  HAS RELATIVE  = overlap >= 0.6 and shared part >= 3 characters
  NO RELATIVE   = everything else
A breakdown by overlap band is printed too.

  python relative_split.py --train ../data/lexicon_split_train.tsv \
      --a val_transformer.tsv --a_name transformer --b val_gru.tsv --b_name gru \
      --flags_out val_relatives.tsv
"""
import argparse
import bisect
import csv

from compare_dumps import load, norm_p, norm_w
from leak_split import leak_key, read_train, report

MIN_RATIO, MIN_CHARS = 0.6, 3
BANDS = [(0.0, 0.4, "< 0.4"), (0.4, 0.6, "0.4-0.6"), (0.6, 0.8, "0.6-0.8"),
         (0.8, 1.0, "0.8-<1"), (1.0, 1.01, "1.0 (whole)")]


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def best_match(key, sorted_keys):
    """Longest common prefix with any key; the best is always a sorted neighbour."""
    i = bisect.bisect_left(sorted_keys, key)
    best, who = 0, ""
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(sorted_keys):
            k = lcp(key, sorted_keys[j])
            if k > best:
                best, who = k, sorted_keys[j]
    return best, who


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--a_pred", default="pred")
    ap.add_argument("--a_name", default="A")
    ap.add_argument("--b", required=True)
    ap.add_argument("--b_pred", default="neural_pred")
    ap.add_argument("--b_name", default="B")
    ap.add_argument("--strip_spaces", action="store_true")
    ap.add_argument("--flags_out")
    a = ap.parse_args()

    train = read_train(a.train)
    fwd = sorted(train)
    rev_map = {k[::-1]: k for k in train}
    rev = sorted(rev_map)

    A, B = load(a.a), load(a.b)
    gold = {norm_w(r["word"]): norm_p(r["gold"], a.strip_spaces) for r in A}
    pa = {norm_w(r["word"]): norm_p(r[a.a_pred], a.strip_spaces) for r in A}
    pb = {norm_w(r["word"]): norm_p(r[a.b_pred], a.strip_spaces) for r in B}
    words = [w for w in gold if w in pb]

    info = {}
    for w in words:
        k = leak_key(w)
        p, pw = best_match(k, fwd)
        s, sw = best_match(k[::-1], rev)
        shared, rel, side = (p, pw, "prefix") if p >= s else (s, rev_map.get(sw, ""), "suffix")
        ratio = shared / max(1, len(k))
        info[w] = (ratio, shared, side, rel)

    has = [w for w in words if info[w][0] >= MIN_RATIO and info[w][1] >= MIN_CHARS]
    no = [w for w in words if w not in set(has)]
    print(f"eval words: {len(words)}   with a relative in train: {len(has)} "
          f"({len(has) / len(words) * 100:.1f}%)   without: {len(no)} "
          f"({len(no) / len(words) * 100:.1f}%)")
    print(f"rule: overlap >= {MIN_RATIO} and >= {MIN_CHARS} shared characters "
          f"(prefix or suffix)")

    report(a.a_name, a.b_name, has, gold, pa, pb, "HAS RELATIVE")
    report(a.a_name, a.b_name, no, gold, pa, pb, "NO RELATIVE")

    print("\nBy overlap band:")
    print(f"  {'band':>10}  {'n':>5}  {a.a_name[:11]:>11}  {a.b_name[:11]:>11}  {'diff':>7}  "
          f"discordant")
    for lo, hi, lab in BANDS:
        sel = [w for w in words if lo <= info[w][0] < hi]
        if not sel:
            continue
        ka = sum(pa[w] == gold[w] for w in sel)
        kb = sum(pb[w] == gold[w] for w in sel)
        oa = sum(pa[w] == gold[w] and pb[w] != gold[w] for w in sel)
        ob = sum(pb[w] == gold[w] and pa[w] != gold[w] for w in sel)
        n = len(sel)
        print(f"  {lab:>10}  {n:5d}  {ka / n:11.3f}  {kb / n:11.3f}  "
              f"{(ka - kb) / n * 100:+6.1f}pp  {oa} vs {ob}")

    if a.flags_out:
        with open(a.flags_out, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(["word", "has_relative", "overlap", "shared_chars", "side",
                         "closest_train_word", "gold", f"{a.a_name}_ok", f"{a.b_name}_ok"])
            for w in words:
                r, sh, side, rel = info[w]
                wr.writerow([w, int(w in set(has)), f"{r:.2f}", sh, side, rel, gold[w],
                             int(pa[w] == gold[w]), int(pb[w] == gold[w])])
        print(f"\nper-word flags written: {a.flags_out}")


if __name__ == "__main__":
    main()
