#!/usr/bin/env python
"""
Blind label audit of the test set: measured label error rate + accuracy against
re-checked labels. The review sheet never shows model output, so the check cannot
favour any model.

STEP 1 - draw the sample (fixed seed, done once):
  python blind_audit.py sample --test ../data/lexicon_split_test.tsv --n 400 --seed 2026 \
      --out ../data/test_audit_sheet.csv

STEP 2 - review the sheet in Excel / LibreOffice. For every row fill `verdict`:
    ok     = gold is correct (and it is the only acceptable pronunciation)
    wrong  = gold is wrong; write the correct pronunciation in `corrected`
    both   = gold is acceptable but so are others; write the others in `corrected`,
             separated by ";"   (e.g. o/O variants that are genuinely both used)
    unsure = cannot decide (reported separately, left out of the rates)
  Write pronunciations in the same phoneme symbols as gold, spaces between phonemes.
  Do NOT open any model dump or prediction file while reviewing.
  Save as CSV (UTF-8).

STEP 3 - score (after the whole sheet is filled):
  python blind_audit.py score --sheet ../data/test_audit_sheet.csv \
      --model transformer=test_tf.tsv:pred --model gru_s1=test_gru_s1.tsv:neural_pred \
      --model gru_s2=test_gru_s2.tsv:neural_pred --model gru_s3=test_gru_s3.tsv:neural_pred
"""
import argparse
import csv
import hashlib
import os
import random
import sys
from collections import Counter

from compare_dumps import load, norm_w, wilson, mcnemar_exact
from canon_rescore import canon, canon_ae
from classify_discordant import classify

FIELDS = ["id", "word", "register", "gold", "verdict", "corrected", "note"]
VERDICTS = {"ok", "wrong", "both", "unsure"}


def cmd_sample(a):
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists - the sample is drawn once. Delete it only if you are sure.")
    rows = []
    with open(a.test, encoding="utf-8-sig") as f:
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) < 2 or p[0].strip().lower().lstrip("#").strip() == "word":
                continue
            if " " in p[0].strip():          # skip sentence rows
                continue
            rows.append((p[0].strip(), p[1].strip(), p[2].strip() if len(p) > 2 else ""))
    rng = random.Random(a.seed)
    pick = sorted(rng.sample(range(len(rows)), a.n))
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:   # BOM: Excel shows Bangla
        w = csv.writer(f)
        w.writerow(FIELDS)
        for k, i in enumerate(pick, 1):
            word, gold, reg = rows[i]
            w.writerow([k, word, reg, canon(gold), "", "", ""])
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"sampled {a.n} of {len(rows)} test words (seed {a.seed}) -> {a.out}")
    print(f"sheet sha256 (record this before reviewing): {h}")


def read_sheet(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [r["id"] for r in rows if r["verdict"].strip().lower() not in VERDICTS]
    if bad:
        sys.exit(f"{len(bad)} rows have no valid verdict (ok/wrong/both/unsure), "
                 f"e.g. ids {bad[:10]}. Finish the review first.")
    miss = [r["id"] for r in rows
            if r["verdict"].strip().lower() in ("wrong", "both") and not r["corrected"].strip()]
    if miss:
        sys.exit(f"rows marked wrong/both need `corrected`: ids {miss[:10]}")
    same = [r["id"] for r in rows if r["verdict"].strip().lower() == "wrong"
            and canon(r["corrected"].split(";")[0]) == canon(r["gold"])]
    if same:
        sys.exit(f"rows marked wrong but `corrected` equals gold: ids {same[:10]}")
    return rows


def cmd_score(a):
    rows = read_sheet(a.sheet)
    n_all = len(rows)
    global canon
    if a.merge_ae:
        canon = canon_ae
        for r in rows:   # a correction that only swaps A/E is no longer an error
            if r["verdict"].strip().lower() == "wrong" and \
                    canon(r["corrected"].split(";")[0]) == canon(r["gold"]):
                r["verdict"] = "ok"
        print("A and E treated as the same sound")
    v = Counter(r["verdict"].strip().lower() for r in rows)
    judged = [r for r in rows if r["verdict"].strip().lower() != "unsure"]
    n = len(judged)

    print(f"sheet: {a.sheet}  rows {n_all}  ok {v['ok']}  wrong {v['wrong']}  "
          f"both {v['both']}  unsure {v['unsure']}")
    lo, hi = wilson(v["wrong"], n)
    print(f"\nLABEL ERROR RATE (wrong / judged): {v['wrong']}/{n} = {v['wrong'] / n:.1%}  "
          f"(95% CI {lo:.1%}-{hi:.1%})")
    lo, hi = wilson(v["both"], n)
    print(f"more than one acceptable pronunciation: {v['both']}/{n} = {v['both'] / n:.1%}  "
          f"(95% CI {lo:.1%}-{hi:.1%})")

    kinds = Counter()
    for r in judged:
        if r["verdict"].strip().lower() == "wrong":
            kinds[classify(canon(r["gold"]), canon(r["corrected"].split(";")[0]))] += 1
    if kinds:
        print("type of label error (gold vs correction):")
        for k, c in kinds.most_common():
            print(f"  {k:<15} {c}")

    # acceptable answers per word
    orig, strict, lenient = {}, {}, {}
    for r in judged:
        w = norm_w(r["word"])
        g = canon(r["gold"])
        alts = [canon(x) for x in r["corrected"].split(";") if x.strip()]
        verdict = r["verdict"].strip().lower()
        orig[w] = {g}
        if verdict == "ok":
            strict[w] = lenient[w] = {g}
        elif verdict == "wrong":
            strict[w] = {alts[0]}
            lenient[w] = set(alts)
        else:  # both
            strict[w] = {g}
            lenient[w] = {g, *alts}

    preds = {}
    for spec in a.model:
        name, rest = spec.split("=", 1)
        path, col = rest.rsplit(":", 1) if ":" in rest else (rest, "pred")
        d = {norm_w(r["word"]): canon(r[col]) for r in load(path)}
        missing = [w for w in orig if w not in d]
        if missing:
            sys.exit(f"{name}: {len(missing)} sampled words not in {path} (e.g. {missing[:3]})")
        preds[name] = d

    words = list(orig)
    print(f"\nMODEL ACCURACY on the {n} judged words (canonical exact match):")
    print(f"  {'model':<14}{'original gold':>22}{'re-checked gold':>24}{'+ alternatives':>24}")
    for name, d in preds.items():
        cells = []
        for ref in (orig, strict, lenient):
            k = sum(d[w] in ref[w] for w in words)
            lo, hi = wilson(k, n)
            cells.append(f"{k / n:6.1%} ({lo:.1%}-{hi:.1%})")
        print(f"  {name:<14}" + "".join(f"{c:>24}" for c in cells))

    names = list(preds)
    if len(names) > 1:
        print("\nPaired tests on re-checked gold (first model vs each other):")
        A = preds[names[0]]
        for nm in names[1:]:
            B = preds[nm]
            oa = sum(A[w] in strict[w] and B[w] not in strict[w] for w in words)
            ob = sum(B[w] in strict[w] and A[w] not in strict[w] for w in words)
            print(f"  {names[0]} vs {nm}: discordant {oa} vs {ob}, "
                  f"McNemar p = {mcnemar_exact(oa, ob):.3g}")

    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(["word", "verdict", "gold", "rechecked", *names])
            for r in judged:
                w = norm_w(r["word"])
                wr.writerow([w, r["verdict"], canon(r["gold"]), " ; ".join(sorted(lenient[w])),
                             *[preds[nm][w] for nm in names]])
        print(f"\nper-word results: {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--test", required=True)
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_sample)
    c = sub.add_parser("score")
    c.add_argument("--sheet", required=True)
    c.add_argument("--model", action="append", required=True,
                   help="name=dump.tsv:prediction_column (repeatable)")
    c.add_argument("--out")
    c.add_argument("--merge_ae", action="store_true",
                   help="treat A and E as the same sound")
    c.set_defaults(func=cmd_score)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()