"""
Detects spelled-out Latin-letter abbreviations (এবিসিডি = A-B-C-D, এলবিডব্লিউ
= L-B-W, এনজিও = N-G-O ...) in the fixed sustlex batch and:

  1. retags them "loanword" (they're not native/tatsama vocabulary --
     REGISTER_OVERRIDES in convert_lexicon.py already does this by hand for
     the one case that came up before, এস; this generalizes it).
  2. forces the leading এ (the letter "A", when the abbreviation starts
     with it) back to closed "e" -- letter-name pronunciation is always
     closed, so any "E" the এ-vowel classifier assigned to one of these
     is wrong; the classifier was trained on organic vocabulary and never
     saw abbreviations.

Detection: greedily match the word against the table of Bangla spellings
for Latin letter names. To avoid false positives against real short native
words that happen to look like a 1-2 letter spelling (এই "this" could
misparse as এ+ই = "A"+"E"), a word only counts as an abbreviation if the
decomposition uses >= 3 letter-units and covers almost the whole word
(a short native suffix like র/টির/দের is allowed to trail).

Usage:
    python fix_abbreviations.py --lexicon sustlex_fixed.tsv --apply
    (omit --apply to preview matches without writing anything)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import g2p_engine as G
from normalizer import normalize

LETTER_NAMES = {
    "A": ["এ"], "B": ["বি"], "C": ["সি"], "D": ["ডি"], "E": ["ই"],
    "F": ["এফ"], "G": ["জি"], "H": ["এইচ", "এচ"], "I": ["আই"], "J": ["জে"],
    "K": ["কে"], "L": ["এল"], "M": ["এম"], "N": ["এন"], "O": ["ও"],
    "P": ["পি"], "Q": ["কিউ"], "R": ["আর"], "S": ["এস"], "T": ["টি"],
    "U": ["ইউ"], "V": ["ভি"], "W": ["ডাব্লিউ", "ডব্লিউ"], "X": ["এক্স"],
    "Y": ["ওয়াই"], "Z": ["জেড", "জ্যাড"],
}
# Longest syllables first so greedy matching doesn't stop early (e.g. ডাব্লিউ before ডব্লিউ).
SYLLABLES = sorted({s for names in LETTER_NAMES.values() for s in names}, key=len, reverse=True)

# Trailing Bangla case/genitive suffixes allowed after the abbreviation itself.
# Deliberately excludes bare "ই" -- that's also the common emphatic particle
# ("X + ই" = "just X"/"X itself"), which attaches to ordinary pronouns and
# words too (একে "to this" + ই = একেই "to this very one"), and made this
# match nearly anything ending in the letter-K syllable কে. Every suffix
# kept here is either multi-character or a case marker (কে/র/এর) that reads
# unambiguously as grammatical case, not emphasis.
TRAILING_SUFFIXES = ["দেরকে", "গুলোর", "গুলোকে", "দের", "গুলো", "টির", "টাকে",
                      "কে", "টির", "টার", "টি", "টা", "র", "এর", "ের", "তে", "য়ের",
                      "তেও", "কেও", "টিও", "টাও", "রও", "এরও", "েরও"]

# Fixed phonemes for each trailing suffix -- these are case/classifier
# markers attached to the END of a larger word, not their own word, so they
# can't be run through the engine independently (it would treat a suffix
# like র as word-INITIAL and wrongly keep a support vowel: "s i" + র ->
# "s i r o" instead of the correct "s i r"). Bangla case suffixes attaching
# to a vowel-final stem never take a support vowel; this is a small closed
# set, not something worth a general rule for.
SUFFIX_PHONEMES = {
    "কে": ["k", "e"], "র": ["r"], "এর": ["e", "r"], "ের": ["e", "r"],
    "টি": ["T", "i"], "টা": ["T", "a"], "টির": ["T", "i", "r"], "টার": ["T", "a", "r"],
    "টাকে": ["T", "a", "k", "e"],
    "দের": ["d", "e", "r"], "দেরকে": ["d", "e", "r", "k", "e"],
    "গুলো": ["g", "u", "l", "O"], "গুলোর": ["g", "u", "l", "O", "r"],
    "গুলোকে": ["g", "u", "l", "O", "k", "e"],
    "তে": ["t", "e"], "য়ের": ["y", "e", "r"],
    "তেও": ["t", "e", "O"], "কেও": ["k", "e", "O"], "টিও": ["T", "i", "O"],
    "টাও": ["T", "a", "O"], "রও": ["r", "O"], "এরও": ["e", "r", "O"], "েরও": ["e", "r", "O"],
}

# Demonstrative pronouns (এ/এই/এটা/এটি "this") + a case marker (কে) and/or a
# particle (ই "emphatic", ও "also/too") decompose cleanly into 3+ letter
# syllables purely by coincidence -- একেই is a real word ("to this very
# one"), not the abbreviation "AKE". Unlike the acronyms this detector is
# built for, these all stop cold at a pronoun + grammatical particle with no
# further letters, so they're pulled by exact match rather than a rule.
EXCLUDE_WORDS = {
    "একেই", "একেও", "এইটি",
    "এটিই", "এটিও", "এটিকে", "এটিকেই", "এটিকেও",
}


def decompose(word):
    """Returns (num_units, chars_matched) for the longest prefix of `word`
    that's a clean run of letter-name syllables, greedy longest-match first."""
    i = 0
    n = len(word)
    units = 0
    while i < n:
        matched = None
        for syl in SYLLABLES:
            if word.startswith(syl, i):
                matched = syl
                break
        if matched is None:
            break
        i += len(matched)
        units += 1
    return units, i


def decompose_units(word):
    """Like decompose(), but returns the actual list of matched syllables."""
    i = 0
    n = len(word)
    out = []
    while i < n:
        matched = None
        for syl in SYLLABLES:
            if word.startswith(syl, i):
                matched = syl
                break
        if matched is None:
            break
        out.append(matched)
        i += len(matched)
    return out


# Most single letter-name syllables (বি "B", টি "T", কে "K", এস "S" ...) are
# also common Bangla grammatical morphemes (case markers, diminutives) or
# fragments of ordinary words, so a bare 2-unit match is too likely to be
# coincidental (টি + কে = টিকে "surviving", not the abbreviation "TK").
# These specific syllables don't have that problem -- they're distinctive
# enough (multi-character, or otherwise not a common Bangla morpheme) that
# they don't double as native grammar, so a 2-unit word is trustworthy as
# long as ONE of its two syllables is one of these anchors (এইচটি "HT",
# এইউ "AU", এইচইউ "HU" are real abbreviations that are only 2 letters).
SAFE_ANCHOR_SYLLABLES = {
    "এইচ", "এচ", "ওয়াই", "ডাব্লিউ", "ডব্লিউ", "এক্স", "জেড", "জ্যাড", "কিউ", "আই", "ইউ",
}


def is_abbreviation(word):
    if word in EXCLUDE_WORDS:
        return False
    parts = decompose_units(word)
    units = len(parts)
    matched_len = sum(len(p) for p in parts)
    if units < 3 and not (units == 2 and any(p in SAFE_ANCHOR_SYLLABLES for p in parts)):
        return False
    remainder = word[matched_len:]
    if remainder == "":
        return True
    for suf in TRAILING_SUFFIXES:
        if remainder == suf:
            return True
    return False


# The generic engine defaults স to "sh" (see phoneme_inventory.py's CONSONANTS
# table) -- correct for native/tatsama words, but wrong for the letter names
# C and S, whose English pronunciation is /s/, not /ʃ/. There's no general
# rule for this in the engine (স -> s is always a per-word lexicon override,
# never a structural default), so it's overridden here explicitly.
LETTER_PHONEME_OVERRIDES = {
    "সি": ["s", "i"],
    "এস": ["e", "s"],
}


def letter_unit_phonemes(unit):
    if unit in LETTER_PHONEME_OVERRIDES:
        return list(LETTER_PHONEME_OVERRIDES[unit])
    return G.text_to_phonemes(normalize(unit))


def rebuild_abbreviation_phonemes(word):
    """
    Re-derives phonemes for an abbreviation by running the engine on each
    letter syllable INDEPENDENTLY and concatenating the results, instead of
    running it on the whole word.

    Why: the engine's schwa-deletion rules assume one continuous native
    word, so a bare consonant mid-abbreviation gets silently dropped if the
    next letter-syllable happens to carry its own vowel sign (Rule 3a/3d).
    That's correct for real Bangla words but wrong here -- letter names are
    each read as their own fixed unit regardless of what follows. Concretely:
    ডব্লিউ ("W") alone gives "D O bl i u", but inside এলবিডব্লিউ ("LBW") the
    engine dropped the O entirely ("D bl i u") because বল্লি carries a vowel.
    Feeding each letter to the engine on its own side-steps this for every
    letter, not just W, since each one gets its own correct word-initial/
    word-final treatment regardless of position in the larger word.
    """
    units, matched_len = decompose(word)
    i = 0
    phonemes = []
    for _ in range(units):
        for syl in SYLLABLES:
            if word.startswith(syl, i):
                phonemes.extend(letter_unit_phonemes(syl))
                i += len(syl)
                break
    remainder = word[i:]
    if remainder:
        phonemes.extend(SUFFIX_PHONEMES[remainder])
    return phonemes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--apply", action="store_true", help="Write changes back to --lexicon")
    args = ap.parse_args()

    with open(args.lexicon, encoding="utf-8") as f:
        lines = f.readlines()

    matches = []
    for i, line in enumerate(lines):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        word, phon, tag = parts
        if is_abbreviation(word):
            matches.append((i, word, phon, tag))

    print(f"Found {len(matches)} abbreviation-shaped words")
    changed = 0
    for i, word, phon, tag in matches:
        new_toks = rebuild_abbreviation_phonemes(word)
        new_phon = " ".join(new_toks)
        new_tag = "loanword"
        if new_phon != phon or new_tag != tag:
            changed += 1
        if args.apply:
            lines[i] = f"{word}\t{new_phon}\t{new_tag}\n"
        else:
            marker = "  <-- CHANGED" if (new_phon != phon or new_tag != tag) else ""
            print(f"  {word}\t{phon} [{tag}]  ->  {new_phon} [{new_tag}]{marker}")

    print(f"\n{changed} of {len(matches)} needed a change (rest already loanword/e)")
    if args.apply:
        with open(args.lexicon, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Wrote changes -> {args.lexicon}")


if __name__ == "__main__":
    main()
