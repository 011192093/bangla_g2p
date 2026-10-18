"""
Keep every derived copy of the lexicon in sync with lexicon.tsv.

lexicon.tsv is the single source of truth. lexicon.tsv.presort.bak and
the three fixed splits (lexicon_split_{train,val,test}.tsv) are frozen
subsets/reorderings of it that do NOT get regenerated on every run (by
design -- val/test accuracy is only meaningful if the same words stay
held out). That means any phoneme/tag fix applied to lexicon.tsv after
a split was generated never reaches the split unless something
explicitly re-syncs it -- which is exactly the drift this script
checks for and fixes.

This only updates phon/tag for words that already exist in a target
file; it never adds or removes words (the split's membership and
presort's row order are untouched -- only their content is corrected).

Usage:
    python sync_splits.py                 # check + fix, print a report
    python sync_splits.py --check_only     # report drift, change nothing
"""
import argparse
import io
import os

TARGETS = [
    "../data/lexicon.tsv.presort.bak",
    "../data/lexicon_split_train.tsv",
    "../data/lexicon_split_val.tsv",
    "../data/lexicon_split_test.tsv",
]


def load(path):
    d = {}
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                d[parts[0]] = (parts[1], parts[2])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="../data/lexicon.tsv")
    ap.add_argument("--check_only", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    lexicon_path = os.path.join(here, args.lexicon)
    source = load(lexicon_path)
    print(f"Source of truth: {args.lexicon} ({len(source)} words)\n")

    any_drift = False
    for rel_path in TARGETS:
        path = os.path.join(here, rel_path)
        if not os.path.exists(path):
            print(f"{rel_path}: not found, skipping")
            continue

        with io.open(path, encoding="utf-8") as f:
            lines = f.readlines()

        n_fixed = 0
        n_malformed = 0
        examples = []
        for i, line in enumerate(lines):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                n_malformed += 1
                continue
            word, phon, tag = parts
            if word in source and source[word] != (phon, tag):
                if len(examples) < 5:
                    examples.append((word, phon, tag, *source[word]))
                if not args.check_only:
                    new_phon, new_tag = source[word]
                    lines[i] = f"{word}\t{new_phon}\t{new_tag}\n"
                n_fixed += 1

        status = "DRIFT" if n_fixed else "clean"
        print(f"{rel_path}: {len(lines)} rows, {n_fixed} stale [{status}]"
              + (f", {n_malformed} malformed row(s)" if n_malformed else ""))
        for w, op, ot, np, nt in examples:
            print(f"    {w}\told=({op}|{ot})\tnew=({np}|{nt})")

        if n_fixed and not args.check_only:
            with io.open(path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"    -> wrote {n_fixed} fixes")

        if n_fixed:
            any_drift = True

    print()
    if args.check_only:
        print("Check-only mode: nothing was written." if any_drift
              else "No drift found.")
    else:
        print("All targets synced." if any_drift else "No drift found -- already in sync.")


if __name__ == "__main__":
    main()
