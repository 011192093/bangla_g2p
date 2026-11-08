#!/usr/bin/env python
"""
Test accuracy with and without the multi-word entries (canonical scoring).
Uses the existing test dumps only; nothing is re-evaluated.

  python multiword_effect.py
"""
from canon_rescore import canon
from compare_dumps import load

DUMPS = [
    ("transformer", "test_tf.tsv", "pred"),
    ("gru_s1", "test_gru_s1.tsv", "neural_pred"),
    ("gru_s2", "test_gru_s2.tsv", "neural_pred"),
    ("gru_s3", "test_gru_s3.tsv", "neural_pred"),
]


def acc(rows, col):
    return sum(canon(r[col]) == canon(r["gold"]) for r in rows) / len(rows)


for name, path, col in DUMPS:
    rows = load(path)
    single = [r for r in rows if " " not in r["word"].strip()]
    multi = [r for r in rows if " " in r["word"].strip()]
    a_all, a_single = acc(rows, col), acc(single, col)
    a_multi = acc(multi, col) if multi else float("nan")
    print(f"{name:<12} all {a_all:.2%} (n={len(rows)})   single words {a_single:.2%} "
          f"(n={len(single)})   multi-word {a_multi:.0%} (n={len(multi)})   "
          f"change {(a_single - a_all) * 100:+.2f} pts")
