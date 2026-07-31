"""
Phase B -- Lexicon Linter
==========================
Runs the same conjunct-pattern / chandrobindu / tatsama-marker checks that
have been applied by hand, batch by batch, all session -- across the WHOLE
lexicon in one pass, so a fix made to word #40 doesn't have to be
rediscovered by hand when word #8000 has the identical bug (as happened
with হ্ন: চিহ্ন already had the right "n n" rendering, but চিহ্নিত and
কাহ্ন silently carried the wrong "n h" for who knows how long, because
every check up to this point was scoped to a specific new-word batch
instead of the full file).

Two tiers:

  Tier 1 (always safe, always applied with --fix): conjunct re-tokenization.
    Strip vowel tokens from both the lexicon entry and the rule engine's
    output; if what's left -- the consonant letters, joined -- is IDENTICAL
    on both sides, the only disagreement is how those letters are grouped
    into tokens (e.g. "k r i" vs "kr i", or "n h" vs "n n" for হ্ন), never
    which sounds are present. That's always safe to re-group to match the
    engine, REGARDLESS of any vowel sitting in between (e.g. lex "a k O n
    d e r" vs engine "a k nd e r" -- the extra O is a separate, legitimate
    question; the n+d -> nd grouping is fixed independently of it).

    Re-grouping only fires when the consonant-token boundaries the engine
    implies land cleanly between lexicon tokens. If a vowel sits INSIDE
    what the engine treats as one atomic multi-consonant conjunct (this
    happens when the word's *real* conjunct is a larger cluster than a
    naive substring match would suggest -- e.g. স্থ্য is a different,
    three-consonant conjunct from স্থ, and only the engine's actual
    grapheme segmentation knows the difference), it's left alone and
    reported under Tier 3 instead of guessed at. An earlier version of
    this script tried to shortcut this with a hand-written list of
    "always safe" conjunct substrings (স্থ, ষ্ণ, স্প, ...) and text-replaced
    them directly -- that broke on স্বাস্থ্য-family words for exactly this
    reason (স্থ্য got matched as if it were স্থ) and produced a fix that
    fought with this same safety check in an infinite loop. Don't
    reintroduce a substring-based shortcut here; always go through the
    engine's real segmentation.

  Tier 3 (report only, never auto-fixed):
    Everything else the rule engine disagrees with the lexicon on --
    mostly legitimate register/vowel-harmony overrides (loanword স -> s,
    schwa retention on inflected/compound words, mid-word ্যা vowel
    quality) that only a person can judge. Printed so they're visible,
    never silently changed.

  Chandrobindu (always applied with --fix):
    word has ঁ but no phoneme token carries "~" -- filled in from the
    engine's nasal placement when it's unambiguous. Alignment is
    consonant-anchored (find_nasal_fixes), not a raw len(toks)==len(eng)
    check -- an earlier version required exact token-count equality and
    exactly one nasal in the engine's output, which silently skipped any
    word with a legitimate extra token on either side (a retained schwa
    in the lexicon, e.g. পাঁচশত with its trailing O) or more than one
    nasal (e.g. আঁকিবুঁকি, which needs two). That version missed 15 words
    lexicon-wide before this was caught by chance on পাঁচশত and swept for.
    Runs after conjunct-regroup so the two consonant skeletons are
    already aligned by the time nasal positions are matched.

  Tag check (always applied with --fix):
    native-tagged words containing ণ/ষ/ৃ/ঞ/ঃ or one of the tatsama
    conjunct clusters (ক্ষ/জ্ঞ/ষ্ণ/ষ্ঠ/ত্ব/ন্ধ/স্ব/শ্ব/দ্ধ/ত্ত্ব/হ্ম) get
    retagged tatsama.

Usage:
    python lint_lexicon.py --lexicon ../data/lexicon.tsv
    python lint_lexicon.py --lexicon ../data/lexicon.tsv --seed ../../data/lexicon_seed.tsv --fix
    python lint_lexicon.py --lexicon ../data/lexicon.tsv --report /path/to/report.txt
"""

import sys
import os
import io
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import g2p_engine as G
from normalizer import normalize

CHANDRABINDU = "ঁ"
VOWEL_TOKENS = set("O o a i u e E A".split())

MARKER_CHARS = set("ণষৃঞঃ")
MARKER_CLUSTERS = ["ত্ব", "ন্ধ", "স্ব", "শ্ব", "দ্ধ", "ত্ত্ব", "হ্ম"]


def is_vowel(tok):
    return tok in VOWEL_TOKENS or (tok.endswith("~") and tok[:-1] in VOWEL_TOKENS)


def is_tatsama_marked(word):
    for ch in word:
        if ch in MARKER_CHARS:
            return True
    for cl in MARKER_CLUSTERS:
        if cl in word:
            return True
    return False


def try_regroup(toks, eng):
    """
    toks, eng: full token lists (vowels included) for the lexicon entry
    and the engine's output. Returns (new_toks, ok). ok=False means the
    consonant letters match but grouping can't be safely re-derived (a
    vowel sits inside an engine-atomic conjunct token) -- caller should
    leave the entry untouched and fall through to a Tier-3 report instead.
    """
    lex_cons = [t for t in toks if not is_vowel(t)]
    eng_cons = [t for t in eng if not is_vowel(t)]
    if "".join(lex_cons) != "".join(eng_cons):
        return toks, False
    if lex_cons == eng_cons:
        return toks, False  # grouping already matches; nothing to regroup

    eng_offsets = []
    pos = 0
    for t in eng_cons:
        eng_offsets.append((pos, pos + len(t), t))
        pos += len(t)

    new_toks = []
    run = []
    run_start = 0
    cursor = 0

    def flush_run():
        nonlocal run
        if not run:
            return True
        run_text = "".join(run)
        start = run_start
        end = start + len(run_text)
        covering = [t for (s, e, t) in eng_offsets if s >= start and e <= end]
        if "".join(covering) != run_text:
            run = []
            return False
        new_toks.extend(covering)
        run = []
        return True

    for tok in toks:
        if is_vowel(tok):
            if not flush_run():
                return toks, False
            new_toks.append(tok)
        else:
            if not run:
                run_start = cursor
            run.append(tok)
            cursor += len(tok)
    if not flush_run():
        return toks, False

    return new_toks, True


def find_nasal_fixes(toks, eng):
    """
    Locate which vowel tokens in `toks` need a "~" appended so their
    nasalization matches the engine's, WITHOUT requiring toks and eng to
    be the same length -- a legitimate extra token on either side (a
    retained schwa in toks, or an extra vowel the engine inserts) is
    common and must not block the check the way an exact len() match
    used to.

    Anchors on consonants: walks both token lists in parallel, and only
    trusts a vowel-to-vowel comparison when both pointers are sitting on
    a vowel at the same point in the shared consonant skeleton. A vowel
    present on only one side is skipped on that side alone, so the
    pointers stay aligned on the next shared consonant. Returns None
    (unsafe, don't touch) if the consonant skeletons don't match at all
    -- that's a genuine conjunct-grouping or consonant-content
    difference, out of scope for this check (falls through to Tier-3
    reporting instead).
    """
    lex_cons = [t for t in toks if not is_vowel(t)]
    eng_cons = [t for t in eng if not is_vowel(t)]
    if lex_cons != eng_cons:
        return None

    to_nasalize = set()
    i, j = 0, 0
    while i < len(toks) and j < len(eng):
        ti, ei = toks[i], eng[j]
        ti_v, ei_v = is_vowel(ti), is_vowel(ei)
        if not ti_v and not ei_v:
            if ti != ei:
                return None  # shouldn't happen given lex_cons == eng_cons, but stay safe
            i += 1
            j += 1
        elif ti_v and ei_v:
            if ei == ti + "~":
                to_nasalize.add(i)
            i += 1
            j += 1
        elif ti_v and not ei_v:
            i += 1  # lex has an extra vowel (e.g. retained schwa); skip it, keep as-is
        else:
            j += 1  # engine has an extra vowel lex doesn't; skip it

    return to_nasalize


def lint_word(word, phon, tag):
    """Returns (new_phon, new_tag, fixes_applied:list[str], flags:list[str])."""
    toks = phon.split()
    fixes = []
    flags = []

    if tag == "native" and is_tatsama_marked(word) and "|" not in phon:
        tag = "tatsama"
        fixes.append("tag:native->tatsama")

    eng = None
    if "|" not in phon:
        try:
            eng = G.text_to_phonemes(normalize(word))
        except Exception:
            eng = None

    if eng is not None and eng != toks:
        new_toks, ok = try_regroup(toks, eng)
        if ok:
            toks = new_toks
            fixes.append("conjunct-regroup")

    if CHANDRABINDU in word and not any(t.endswith("~") for t in toks) and eng is not None:
        nasal_fixes = find_nasal_fixes(toks, eng)
        if nasal_fixes:
            for idx in nasal_fixes:
                toks[idx] = toks[idx] + "~"
            fixes.append("chandrobindu")

    if eng is not None and eng != toks:
        lex_cons = [t for t in toks if not is_vowel(t)]
        eng_cons = [t for t in eng if not is_vowel(t)]
        kind = "consonant-diff" if "".join(lex_cons) != "".join(eng_cons) else "vowel-only-diff"
        flags.append(f"{kind}: lex={toks} eng={eng}")

    return " ".join(toks), tag, fixes, flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", required=True, help="Path to lexicon.tsv (word, phonemes, tag)")
    ap.add_argument("--seed", default=None, help="Optional lexicon_seed.tsv to mirror phoneme fixes into")
    ap.add_argument("--fix", action="store_true", help="Write fixes back to --lexicon (and --seed if given). Without this flag, dry-run only.")
    ap.add_argument("--report", default=None, help="Optional path to write the full Tier-3 flag list (consonant-diffs only, the ones worth a human look)")
    args = ap.parse_args()

    with io.open(args.lexicon, encoding="utf-8") as f:
        lines = f.readlines()

    total = 0
    fix_counts = {}
    flag_consonant = []
    flag_vowel_only = 0
    changed_words = {}

    for i, line in enumerate(lines):
        raw = line.rstrip("\n")
        parts = raw.split("\t")
        if len(parts) != 3:
            continue
        word, phon, tag = parts
        total += 1
        new_phon, new_tag, fixes, flags = lint_word(word, phon, tag)

        for f_ in fixes:
            fix_counts[f_] = fix_counts.get(f_, 0) + 1
        for fl in flags:
            if fl.startswith("consonant-diff"):
                flag_consonant.append((i + 1, word, phon, tag, fl))
            else:
                flag_vowel_only += 1

        if new_phon != phon or new_tag != tag:
            changed_words[word] = new_phon
            if args.fix:
                lines[i] = f"{word}\t{new_phon}\t{new_tag}\n"

    print(f"Scanned {total} entries")
    print(f"Fixes {'applied' if args.fix else 'that WOULD be applied with --fix'}:")
    for k, v in sorted(fix_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print(f"Tier-3 flags (never auto-fixed): {len(flag_consonant)} consonant-diff, {flag_vowel_only} vowel-only (expected/legitimate, not printed)")

    if args.fix:
        with io.open(args.lexicon, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"\nWrote fixes to {args.lexicon}")

        if args.seed:
            with io.open(args.seed, encoding="utf-8") as f:
                seed_lines = f.readlines()
            updated = 0
            for i, line in enumerate(seed_lines):
                parts = line.rstrip("\n").split("\t")
                if parts and parts[0] in changed_words:
                    want = changed_words[parts[0]]
                    if len(parts) < 2 or parts[1] != want:
                        new_parts = [parts[0], want] + parts[2:]
                        seed_lines[i] = "\t".join(new_parts) + "\n"
                        updated += 1
            with io.open(args.seed, "w", encoding="utf-8") as f:
                f.writelines(seed_lines)
            print(f"Mirrored {updated} phoneme fixes into {args.seed}")

    if args.report:
        with io.open(args.report, "w", encoding="utf-8") as f:
            f.write(f"Tier-3 consonant-diff flags: {len(flag_consonant)}\n\n")
            for line_no, word, phon, tag, fl in flag_consonant:
                f.write(f"line {line_no}: {word}\t{phon}\t[{tag}]\t{fl}\n")
        print(f"Wrote Tier-3 report to {args.report}")


if __name__ == "__main__":
    main()
