#!/usr/bin/env python
"""
Where did each audited label error come from: the source respelling, or the conversion?

STEP 1 - make the sheet (the 'wrong' rows of the audit + the source respelling):
  python error_source.py make --sheet ../data/test_audit_sheet.csv \
      --source PATH_TO_SUSTLEX.tsv --out ../data/error_source_sheet.csv
  (source file: word<TAB>Bangla-script respelling, one per line; other columns ignored)

STEP 2 - open the CSV and fill `source_of_error` for each row:
  source     = the respelling itself is wrong (the conversion followed it faithfully)
  conversion = the respelling is right (or allows the right reading), the phonemes are not
  later      = the gold did not come from the conversion (seed lexicon or a later hand edit)
  both       = respelling and conversion are both wrong
  unclear    = respelling is ambiguous (e.g. inherent vowel not marked) - cannot tell

Set several rows at once (writes a new sheet, original untouched):
  python error_source.py set --sheet ../data/error_source_sheet.csv \
      --ids 8,208,339 --value later --out ../data/error_source_sheet_v2.csv

STEP 3 - summary:
  python error_source.py summary --sheet ../data/error_source_sheet.csv
"""
import argparse
import csv
import sys
import unicodedata
from collections import Counter, defaultdict

from canon_rescore import canon
from classify_discordant import classify
from compare_dumps import wilson

NUKTA = {"য়": "য়", "ড়": "ড়", "ঢ়": "ঢ়"}
CHOICES = {"source", "conversion", "later", "both", "unclear"}


def norm(w):
    w = unicodedata.normalize("NFC", w.strip())
    for a, b in NUKTA.items():
        w = w.replace(a, b)
    return w


def cmd_make(a):
    src = {}
    with open(a.source, encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            p = line.split("\t") if "\t" in line else line.split()   # SUSTlex: space-separated
            if len(p) >= 2 and p[0].strip():
                src.setdefault(norm(p[0]), p[1].strip())
    with open(a.sheet, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["verdict"].strip().lower() == "wrong"]
    missing = 0
    with open(a.out, "x", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "word", "register", "source_respelling", "gold", "corrected",
                    "error_type", "note", "source_of_error"])
        for r in rows:
            resp = src.get(norm(r["word"]), "")
            missing += not resp
            w.writerow([r["id"], r["word"], r["register"], resp, r["gold"], r["corrected"],
                        classify(canon(r["gold"]), canon(r["corrected"])), r["note"], ""])
    print(f"{len(rows)} wrong-label rows written to {a.out}; "
          f"{missing} words not found in the source (check spelling fixes)")


def cmd_summary(a):
    with open(a.sheet, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [r["id"] for r in rows if r["source_of_error"].strip().lower() not in CHOICES]
    if bad:
        sys.exit(f"fill source_of_error ({'/'.join(sorted(CHOICES))}) for ids {bad[:10]}")
    n = len(rows)
    c = Counter(r["source_of_error"].strip().lower() for r in rows)
    print(f"label errors: {n}")
    for k in ("source", "conversion", "later", "both", "unclear"):
        lo, hi = wilson(c[k], n)
        print(f"  {k:<11} {c[k]:3d}  {c[k] / n:6.1%}  (95% CI {lo:.1%}-{hi:.1%})")
    by = defaultdict(Counter)
    for r in rows:
        by[r["error_type"]][r["source_of_error"].strip().lower()] += 1
    print("\nby error type:")
    print(f"  {'type':<15}{'source':>8}{'conversion':>11}{'later':>7}{'both':>6}{'unclear':>9}")
    for t, cc in sorted(by.items(), key=lambda x: -sum(x[1].values())):
        print(f"  {t:<15}{cc['source']:>8}{cc['conversion']:>11}{cc['later']:>7}{cc['both']:>6}"
              f"{cc['unclear']:>9}")


def cmd_set(a):
    if a.value not in CHOICES:
        sys.exit(f"value must be one of {sorted(CHOICES)}")
    ids = {x.strip() for x in a.ids.split(",") if x.strip()}
    with open(a.sheet, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    found = set()
    for r in rows:
        if r["id"].strip() in ids:
            r["source_of_error"] = a.value
            found.add(r["id"].strip())
    missing = ids - found
    if missing:
        sys.exit(f"ids not in sheet: {sorted(missing)} - nothing written")
    with open(a.out, "x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"set {len(found)} rows to '{a.value}' -> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--sheet", required=True)
    m.add_argument("--source", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_make)
    t = sub.add_parser("set")
    t.add_argument("--sheet", required=True)
    t.add_argument("--ids", required=True)
    t.add_argument("--value", required=True)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_set)
    s = sub.add_parser("summary")
    s.add_argument("--sheet", required=True)
    s.set_defaults(func=cmd_summary)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
