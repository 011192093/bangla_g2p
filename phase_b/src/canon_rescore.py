#!/usr/bin/env python
"""
Canonical phoneme tokenization: same sound, same tokens.

The lexicon writes the same pronunciation with different token boundaries, e.g.
  m O n t r i   vs  m O nt ri       (মন্ত্রি)
  p r i         vs  p ri / pr i     (প্রি)
  k kh          vs  kkh             (ক্ষ)
  s th          vs  sth             (স্থ)
Exact match counts these as errors although the pronunciation is identical.

canon() splits every token into atomic phonemes (greedy longest match against ATOMS),
so "nt ri" and "n t r i" both become "n t r i". It never changes a sound, only spacing.
Apply it identically to gold and predictions at scoring time (and, if you decide, to the
train labels before retraining). Fix it BEFORE the final test run.

  # rescore two dumps raw vs canonical
  python canon_rescore.py rescore --a val_transformer.tsv --a_name transformer --b val_gru.tsv --b_name gru

  # how inconsistent is a label file?
  python canon_rescore.py audit --file ../data/lexicon_split_train.tsv

  # write a canonicalized copy of a label file (new file, original untouched)
  python canon_rescore.py write --file ../data/lexicon_split_train.tsv --out ../data/train_canon.tsv
"""
import argparse
import csv
from collections import Counter

from compare_dumps import load, norm_w, wilson, mcnemar_exact

# Atomic phonemes, longest first. Multi-letter atoms are single sounds.
ATOMS = sorted([
    "chh", "kh", "gh", "ch", "jh", "Th", "Dh", "th", "dh", "ph", "bh", "sh", "ng", "ny",
    "a", "A", "e", "E", "i", "o", "O", "u",
    "k", "g", "c", "j", "T", "D", "t", "d", "n", "p", "b", "m", "r", "l", "s", "h",
    "y", "w", "R", "f", "v", "z", "x", "q", "~", "|",
], key=len, reverse=True)


def split_token(tok):
    out, i = [], 0
    while i < len(tok):
        for a in ATOMS:
            if tok.startswith(a, i):
                out.append(a)
                i += len(a)
                break
        else:  # unknown character: keep it as its own unit
            out.append(tok[i])
            i += 1
    return out


def canon(p):
    return " ".join(t for tok in p.split() for t in split_token(tok))


def cmd_rescore(a):
    A, B = load(a.a), load(a.b)
    gold = {norm_w(r["word"]): r["gold"] for r in A}
    pa = {norm_w(r["word"]): r[a.a_pred] for r in A}
    pb = {norm_w(r["word"]): r[a.b_pred] for r in B}
    words = [w for w in gold if w in pb]
    n = len(words)
    sq = lambda s: " ".join(s.split())
    for label, f in (("RAW", sq), ("CANONICAL", canon)):
        ka = sum(f(pa[w]) == f(gold[w]) for w in words)
        kb = sum(f(pb[w]) == f(gold[w]) for w in words)
        oa = sum(f(pa[w]) == f(gold[w]) != f(pb[w]) for w in words)
        ob = sum(f(pb[w]) == f(gold[w]) != f(pa[w]) for w in words)
        print(f"\n[{label}]  n={n}")
        for nm, k in ((a.a_name, ka), (a.b_name, kb)):
            lo, hi = wilson(k, n)
            print(f"  {nm:>12}: {k}/{n} = {k / n:.4f}  (95% CI {lo:.4f}-{hi:.4f})")
        print(f"  difference: {(ka - kb) / n * 100:+.2f} pp   discordant {oa} vs {ob}   "
              f"McNemar p = {mcnemar_exact(oa, ob):.4g}")
    for nm, p in ((a.a_name, pa), (a.b_name, pb)):
        fixed = sum(sq(p[w]) != sq(gold[w]) and canon(p[w]) == canon(gold[w]) for w in words)
        print(f"{nm}: {fixed} 'errors' were only token spacing")


def cmd_audit(a):
    cl = Counter()
    rows = changed = 0
    with open(a.file, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2 or not parts[1].strip() or \
                    parts[0].strip().lower().lstrip("#").strip() == "word":
                continue
            rows += 1
            toks = parts[1].split()
            if canon(parts[1]) != " ".join(toks):
                changed += 1
            for t in toks:
                if len(split_token(t)) > 1:
                    cl[t] += 1
    print(f"rows: {rows}   rows using any cluster token: {changed} ({changed / rows * 100:.1f}%)")
    print("most common cluster tokens (each also appears split elsewhere unless noted):")
    for t, c in cl.most_common(a.top):
        print(f"  {t:>6} -> {' '.join(split_token(t)):<10} {c}")


def cmd_write(a):
    with open(a.file, encoding="utf-8-sig") as f, \
            open(a.out, "x", encoding="utf-8", newline="") as g:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2 and parts[1].strip() and not parts[0].lower().startswith(
                    ("word", "#")):
                parts[1] = canon(parts[1])
            g.write("\t".join(parts) + "\n")
    print(f"written {a.out} (original untouched)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rescore")
    r.add_argument("--a", required=True)
    r.add_argument("--a_pred", default="pred")
    r.add_argument("--a_name", default="A")
    r.add_argument("--b", required=True)
    r.add_argument("--b_pred", default="neural_pred")
    r.add_argument("--b_name", default="B")
    r.set_defaults(func=cmd_rescore)
    u = sub.add_parser("audit")
    u.add_argument("--file", required=True)
    u.add_argument("--top", type=int, default=40)
    u.set_defaults(func=cmd_audit)
    w = sub.add_parser("write")
    w.add_argument("--file", required=True)
    w.add_argument("--out", required=True)
    w.set_defaults(func=cmd_write)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
