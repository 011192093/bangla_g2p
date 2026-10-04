"""
Generates a permanent, stratified 94/3/3 train/val/test split of the lexicon
and writes it to three fixed files. Run this ONCE; train.py then loads the
three files directly instead of re-splitting the lexicon on every run.

Stratified by register tag so train/val/test each get a proportional share
of native/loanword/tatsama/proper_noun/etc, rather than letting a rare tag
end up entirely in one split by chance.

Usage:
    python make_fixed_split.py --lexicon ../data/lexicon.tsv --out_dir ../data --seed 42
"""
import os
import argparse
import random
from collections import defaultdict


def load_lexicon(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            entries.append(tuple(parts))  # (word, phonemes, tag)
    return entries


def stratified_split(entries, train_ratio, val_ratio, test_ratio, seed):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    by_tag = defaultdict(list)
    for e in entries:
        by_tag[e[2]].append(e)

    rng = random.Random(seed)
    train, val, test = [], [], []

    for tag, rows in sorted(by_tag.items()):
        rows = rows[:]
        rng.shuffle(rows)
        n = len(rows)
        n_val = max(1, round(n * val_ratio)) if n >= 3 else 0
        n_test = max(1, round(n * test_ratio)) if n >= 3 else 0
        # keep at least everything in train for tags too small to split meaningfully
        if n_val + n_test >= n:
            train.extend(rows)
            continue
        val.extend(rows[:n_val])
        test.extend(rows[n_val:n_val + n_test])
        train.extend(rows[n_val + n_test:])

    # shuffle each split's row order once more so they aren't grouped by tag
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_split(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for word, phon, tag in rows:
            f.write(f"{word}\t{phon}\t{tag}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="../data/lexicon.tsv")
    ap.add_argument("--out_dir", default="../data")
    ap.add_argument("--train_ratio", type=float, default=0.94)
    ap.add_argument("--val_ratio", type=float, default=0.03)
    ap.add_argument("--test_ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    entries = load_lexicon(args.lexicon)
    print(f"Loaded {len(entries)} entries from {args.lexicon}")

    train, val, test = stratified_split(
        entries, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    train_path = os.path.join(args.out_dir, "lexicon_split_train.tsv")
    val_path = os.path.join(args.out_dir, "lexicon_split_val.tsv")
    test_path = os.path.join(args.out_dir, "lexicon_split_test.tsv")

    write_split(train_path, train)
    write_split(val_path, val)
    write_split(test_path, test)

    total = len(train) + len(val) + len(test)
    print(f"Train: {len(train)} ({100*len(train)/total:.2f}%) -> {train_path}")
    print(f"Val:   {len(val)} ({100*len(val)/total:.2f}%) -> {val_path}")
    print(f"Test:  {len(test)} ({100*len(test)/total:.2f}%) -> {test_path}")

    # sanity check: no leakage (same word+phon in two splits), and print per-tag breakdown
    def tag_counts(rows):
        c = defaultdict(int)
        for _, _, tag in rows:
            c[tag] += 1
        return c

    print("\nPer-tag counts (train / val / test):")
    all_tags = sorted(set(e[2] for e in entries))
    tc_train, tc_val, tc_test = tag_counts(train), tag_counts(val), tag_counts(test)
    for tag in all_tags:
        print(f"  {tag:20s} {tc_train.get(tag,0):7d} / {tc_val.get(tag,0):5d} / {tc_test.get(tag,0):5d}")

    train_set = set((w, p) for w, p, _ in train)
    val_set = set((w, p) for w, p, _ in val)
    test_set = set((w, p) for w, p, _ in test)
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set
    print(f"\nExact (word,phon) overlap -- train/val: {len(overlap_tv)}  train/test: {len(overlap_tt)}  val/test: {len(overlap_vt)}")


if __name__ == "__main__":
    main()
