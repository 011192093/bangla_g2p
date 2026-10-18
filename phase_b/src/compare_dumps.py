#!/usr/bin/env python
"""
Paired comparison of two G2P models on the same eval file (exact McNemar test).

Reads two dump TSVs with a header row. For each, give the column holding the prediction.
Gold is taken from file A's `gold` column; words are matched after norm_bn.

  python compare_dumps.py --a fresh_val_transformer.tsv --a_pred pred --a_name transformer \
      --b fresh_val_gru.tsv --b_pred neural_pred --b_name gru

Also prints a confidence-bucket table for any file that has a `confidence` column.
"""
import argparse
import csv
import math
import unicodedata

NUKTA_FIX = {"য়": "য়", "ড়": "ড়", "ঢ়": "ঢ়"}


def norm_w(w):
    w = unicodedata.normalize("NFC", w.strip())
    for a, b in NUKTA_FIX.items():
        w = w.replace(a, b)
    return w


def norm_p(p, strip):
    p = " ".join(p.split())
    return p.replace(" ", "") if strip else p


GRU_COLS = ["word", "gold", "rule_pred", "neural_pred", "confidence"]
ALIASES = {"grapheme": "word", "graphemes": "word", "input": "word",
           "reference": "gold", "ref": "gold", "target": "gold",
           "neural": "neural_pred", "model_pred": "neural_pred", "prediction": "pred",
           "rule": "rule_pred", "conf": "confidence"}


def load(path):
    """Reads a dump with or without a header; column names are matched loosely."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if any(c.strip() for c in r)]
    head = [c.strip().lstrip("#").strip().lower().replace(" ", "_") for c in rows[0]]
    head = [ALIASES.get(c, c) for c in head]
    if "word" in head:
        cols, data = head, rows[1:]
    else:  # no header: assume compare_with_engine.py column order
        cols, data = GRU_COLS[:len(rows[0])], rows
        print(f"note: {path} has no recognised header (first line: {rows[0][:5]}); "
              f"assuming columns {cols}")
    return [dict(zip(cols, r + [""] * (len(cols) - len(r)))) for r in data]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def conf_table(rows, pred_col, gold, strip, name):
    if not rows or "confidence" not in rows[0]:
        return
    buckets = [(0.99, 1.01, ">= 0.99"), (0.97, 0.99, "0.97-0.99"), (0.95, 0.97, "0.95-0.97"),
               (0.90, 0.95, "0.90-0.95"), (-1, 0.90, "< 0.90")]
    print(f"\nConfidence vs accuracy ({name}):")
    for lo, hi, lab in buckets:
        sel = [r for r in rows if _f(r.get("confidence")) is not None
               and lo <= _f(r["confidence"]) < hi]
        if sel:
            k = sum(norm_p(r[pred_col], strip) == gold[norm_w(r["word"])] for r in sel)
            print(f"  {lab:>10}  n={len(sel):5d}  acc={k / len(sel):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--a_pred", default="pred")
    ap.add_argument("--a_name", default="A")
    ap.add_argument("--b", required=True)
    ap.add_argument("--b_pred", default="neural_pred")
    ap.add_argument("--b_name", default="B")
    ap.add_argument("--strip_spaces", action="store_true",
                    help="compare phoneme strings with all spaces removed "
                         "(use if the two dumps space phonemes differently)")
    ap.add_argument("--out", help="optional TSV of the discordant words")
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    gold = {norm_w(r["word"]): norm_p(r["gold"], a.strip_spaces) for r in A}
    pa = {norm_w(r["word"]): norm_p(r[a.a_pred], a.strip_spaces) for r in A}
    pb = {norm_w(r["word"]): norm_p(r[a.b_pred], a.strip_spaces) for r in B}
    gb = {norm_w(r["word"]): norm_p(r["gold"], a.strip_spaces) for r in B if "gold" in r}

    words = [w for w in gold if w in pb]
    missing = len(gold) - len(words)
    gold_diff = sum(1 for w in words if w in gb and gb[w] != gold[w])
    n = len(words)
    ka = sum(pa[w] == gold[w] for w in words)
    kb = sum(pb[w] == gold[w] for w in words)
    only_a = [w for w in words if pa[w] == gold[w] and pb[w] != gold[w]]
    only_b = [w for w in words if pb[w] == gold[w] and pa[w] != gold[w]]

    print(f"matched words: {n}  (in A but not B: {missing})")
    if gold_diff:
        print(f"WARNING: {gold_diff} words have different gold in the two dumps "
              f"(different label versions or spacing). Results use A's gold.")
    for name, k in ((a.a_name, ka), (a.b_name, kb)):
        lo, hi = wilson(k, n)
        print(f"{name:>12}: {k}/{n} = {k / n:.4f}  (95% CI {lo:.4f}-{hi:.4f})")
    print(f"difference ({a.a_name} - {a.b_name}): {(ka - kb) / n * 100:+.2f} pp")
    print(f"discordant: only {a.a_name} right = {len(only_a)}, "
          f"only {a.b_name} right = {len(only_b)}")
    print(f"exact McNemar p = {mcnemar_exact(len(only_a), len(only_b)):.4g}")

    conf_table(A, a.a_pred, gold, a.strip_spaces, a.a_name)
    conf_table(B, a.b_pred, gold, a.strip_spaces, a.b_name)

    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(["word", "gold", f"{a.a_name}_pred", f"{a.b_name}_pred", "who_right"])
            for w in only_a:
                wr.writerow([w, gold[w], pa[w], pb[w], a.a_name])
            for w in only_b:
                wr.writerow([w, gold[w], pa[w], pb[w], a.b_name])
        print(f"\ndiscordant words written: {a.out}")


if __name__ == "__main__":
    main()
