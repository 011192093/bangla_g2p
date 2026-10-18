"""
Merge the reviewed 600-word fresh-val draft with the 87-word replacement
batch (the words that originally contained a decomposed য়/ড়/ঢ় and so
were never actually seen by the rule engine, the neural model, or the
lexicon -- see normalizer.compose_nukta).

Drops the 87 decomposed-spelling rows from the reviewed600 file and
appends the clean replacement rows in their place, so the result is
still exactly 600 words, now with zero decomposed-nukta entries.

--reviewed600 accepts either the finalized 3-column format (word, phonemes,
register -- what the current lexicon_fresh_val.tsv already is, including
every later hand-correction made after the original draft) or the original
8-column review format; detected automatically by column count.

--replacement is always the 8-column review format:
    word  draft_phonemes  register  rule_phonemes  draft_source
    verdict(ok/fix)  corrected_phonemes  corrected_register

Any row in --replacement whose verdict/corrected_phonemes/corrected_register
are still blank gets resolved here with the same policy used throughout
this project: exact match or safe conjunct-regroup (lint_lexicon.try_regroup)
against rule_phonemes wins automatically; otherwise, since draft_source is
"converter" (pulled from the already-reviewed lexicon, not a fresh engine
guess), the draft is trusted over the naive rule engine.

Output is always the finalized 3-column format (word, phonemes, register).

Usage:
    python merge_fresh_val.py --reviewed600 fresh_val_draft_reg.tsv \
        --replacement fresh_val_replacement87.tsv --out fresh_val_merged.tsv
"""
import argparse
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from lint_lexicon import try_regroup
from normalizer import compose_nukta

NUKTA_DECOMPOSED = ("য়", "ড়", "ঢ়")  # য+়, ড+়, ঢ+়


def has_decomposed_nukta(word):
    return any(seq in word for seq in NUKTA_DECOMPOSED)


def read_review_tsv(path):
    """Always reads as the 8-column review format, padding short rows."""
    with io.open(path, encoding="utf-8") as f:
        lines = f.readlines()
    header = lines[0]
    rows = []
    for line in lines[1:]:
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        while len(parts) < 8:
            parts.append("")
        rows.append(parts)
    return header, rows


def read_reviewed600(path):
    """Reads --reviewed600, auto-detecting 3-column (finalized) vs
    8-column (review-format) input. Returns a list of (word, phon, tag)."""
    with io.open(path, encoding="utf-8") as f:
        lines = f.readlines()
    first_data_line = next((l for l in lines if l.strip() and not l.startswith("word\t")), "")
    n_cols = len(first_data_line.rstrip("\r\n").split("\t"))

    if n_cols == 3:
        triples = []
        for line in lines:
            line = line.rstrip("\r\n")
            if not line.strip() or line.startswith("word\t"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            triples.append(tuple(parts))
        return triples, "3col"
    else:
        _, rows = read_review_tsv(path)
        resolved = [resolve_row(r) for r in rows]
        return [(r[0], r[6], r[7]) for r in resolved], "8col"


def resolve_row(parts):
    """Fill verdict/corrected_phonemes/corrected_register if blank."""
    word, draft_phon, register, rule_phon, draft_source = (p.strip() for p in parts[:5])
    verdict, corrected_phon, corrected_reg = parts[5].strip(), parts[6].strip(), parts[7].strip()

    if corrected_phon and corrected_reg:
        return parts  # already resolved, leave as-is

    draft_toks = draft_phon.split()
    rule_toks = rule_phon.split()

    if draft_toks == rule_toks:
        verdict = "ok"
        corrected_phon = draft_phon
    else:
        new_toks, ok = try_regroup(draft_toks, rule_toks)
        if ok and new_toks != draft_toks:
            verdict = "fix"
            corrected_phon = " ".join(new_toks)
        else:
            # draft_source == "converter" for every row in this batch --
            # trust the reviewed draft over the naive rule engine.
            verdict = "ok"
            corrected_phon = draft_phon

    corrected_reg = corrected_reg or register
    return parts[:5] + [verdict, corrected_phon, corrected_reg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewed600", required=True)
    ap.add_argument("--replacement", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reviewed_triples, reviewed_fmt = read_reviewed600(args.reviewed600)
    print(f"Reviewed600 ({reviewed_fmt}): {len(reviewed_triples)} rows loaded")

    kept = [t for t in reviewed_triples if not has_decomposed_nukta(t[0])]
    dropped = [t for t in reviewed_triples if has_decomposed_nukta(t[0])]
    print(f"  -> dropping {len(dropped)} decomposed-nukta rows, keeping {len(kept)}")

    _, replacement_rows = read_review_tsv(args.replacement)
    resolved_replacement = [resolve_row(r) for r in replacement_rows]
    replacement_triples = []
    for r in resolved_replacement:
        word = compose_nukta(r[0])  # defensive: in case a hand-typed row slipped in decomposed
        if has_decomposed_nukta(word):
            print(f"  WARNING: replacement word still decomposed after compose_nukta: {word!r}")
        replacement_triples.append((word, r[6], r[7]))
    print(f"Replacement: {len(replacement_triples)} rows, all resolved")

    merged = kept + replacement_triples
    print(f"Merged total: {len(merged)} rows "
          f"(expected 600: {'OK' if len(merged) == 600 else 'MISMATCH'})")

    with io.open(args.out, "w", encoding="utf-8") as f:
        for word, phon, tag in merged:
            f.write(f"{word}\t{phon}\t{tag}\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
