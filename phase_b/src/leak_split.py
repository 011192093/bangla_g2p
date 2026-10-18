#!/usr/bin/env python
"""
Leakage check: is a model's val gain coming from near-duplicates of train words?

A val word counts as LEAKED if, after the fixed normalization below, its spelling equals
the normalized spelling of any train word (e.g. সেনাবাহিনী in val, সেনাবাহিনি in train).
The rule only looks at spellings, never at model output, so it is safe to also run on test
later. Decide and freeze the rule BEFORE looking at the numbers.

  python leak_split.py --train ../data/lexicon_split_train.tsv \
      --a val_transformer.tsv --a_name transformer --b val_gru.tsv --b_name gru \
      --flags_out val_leaked.tsv
"""
import argparse
import csv
import hashlib
import unicodedata

from compare_dumps import load, norm_p, norm_w, wilson, mcnemar_exact

# Fixed rule (version 1). Same-sound letter merges, as in Ahmad et al. 2019 Sect. 4.2.
MERGE = {
    "ী": "ি",  # ী -> ি
    "ঈ": "ই",  # ঈ -> ই
    "ূ": "ু",  # ূ -> ু
    "ঊ": "উ",  # ঊ -> উ
    "ষ": "শ",  # ষ -> শ
    "ণ": "ন",  # ণ -> ন
}


def leak_key(w):
    w = norm_w(w)
    for a, b in MERGE.items():
        w = w.replace(a, b)
    return w


def read_train(path):
    words = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2 and parts[0].strip():
                words.setdefault(leak_key(parts[0]), []).append((parts[0].strip(),
                                                                 parts[1].strip()))
    return words


def report(name_a, name_b, words, gold, pa, pb, label):
    n = len(words)
    if n == 0:
        print(f"\n[{label}] no words")
        return
    ka = sum(pa[w] == gold[w] for w in words)
    kb = sum(pb[w] == gold[w] for w in words)
    oa = sum(pa[w] == gold[w] and pb[w] != gold[w] for w in words)
    ob = sum(pb[w] == gold[w] and pa[w] != gold[w] for w in words)
    print(f"\n[{label}]  n={n}")
    for nm, k in ((name_a, ka), (name_b, kb)):
        lo, hi = wilson(k, n)
        print(f"  {nm:>12}: {k}/{n} = {k / n:.4f}  (95% CI {lo:.4f}-{hi:.4f})")
    print(f"  difference: {(ka - kb) / n * 100:+.2f} pp   discordant {oa} vs {ob}   "
          f"McNemar p = {mcnemar_exact(oa, ob):.4g}")


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
    ap.add_argument("--flags_out", help="TSV listing each eval word and its leak flag")
    a = ap.parse_args()

    train = read_train(a.train)
    A, B = load(a.a), load(a.b)
    gold = {norm_w(r["word"]): norm_p(r["gold"], a.strip_spaces) for r in A}
    pa = {norm_w(r["word"]): norm_p(r[a.a_pred], a.strip_spaces) for r in A}
    pb = {norm_w(r["word"]): norm_p(r[a.b_pred], a.strip_spaces) for r in B}
    words = [w for w in gold if w in pb]

    exact = [w for w in words if any(norm_w(t) == w for t, _ in train.get(leak_key(w), []))]
    leaked = [w for w in words if leak_key(w) in train]
    clean = [w for w in words if leak_key(w) not in train]

    print(f"train keys: {len(train)}   eval words: {len(words)}")
    print(f"leaked (normalized spelling matches a train word): {len(leaked)} "
          f"({len(leaked) / len(words) * 100:.1f}%)   of which exact string matches: "
          f"{len(exact)}")

    same_pron = sum(1 for w in leaked
                    if any(norm_p(p, a.strip_spaces) == gold[w] for _, p in train[leak_key(w)]))
    print(f"leaked words whose train twin has the SAME gold: {same_pron}/{len(leaked)}  "
          f"(different gold = possible label inconsistency; fix in train only)")

    report(a.a_name, a.b_name, words, gold, pa, pb, "ALL")
    report(a.a_name, a.b_name, leaked, gold, pa, pb, "LEAKED")
    report(a.a_name, a.b_name, clean, gold, pa, pb, "CLEAN")

    if a.flags_out:
        with open(a.flags_out, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(["word", "leaked", "train_twins", "gold", "twin_golds"])
            for w in words:
                tw = train.get(leak_key(w), [])
                wr.writerow([w, int(bool(tw)), " | ".join(t for t, _ in tw), gold[w],
                             " | ".join(p for _, p in tw)])
        h = hashlib.sha256(open(a.flags_out, "rb").read()).hexdigest()
        print(f"\nflags written: {a.flags_out}  sha256={h}")


if __name__ == "__main__":
    main()
