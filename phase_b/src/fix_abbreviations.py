"""
Detect and fix English-letter-name abbreviations in lexicon.tsv (CRPF, NGO,
BNP, DC, MP, HIV, ATM, ...). These are built by concatenating a closed set of
~26 Bangla spellings of English letter names (সি=C, আর=R, পি=P, এফ=F, ...).
Today they're scattered across native/loanword/proper_noun with inconsistent,
often wrong phonemes (e.g. সি pronounced "sh i" instead of "s i").

A word is treated as an abbreviation iff it decomposes EXACTLY into a
sequence of >=2 known letter tokens with nothing left over (longest-match
tokenizer). Matches get:
    phonemes = concatenation of each letter's canonical phoneme
    tag      = "abbreviation" (new register tag)

Usage:
    python fix_abbreviations.py              # dry run, report only
    python fix_abbreviations.py --write      # apply to lexicon.tsv
"""
import argparse
import io

# Bangla spelling(s) -> canonical phoneme, for each English letter name.
# Ground-truthed against this lexicon's own standalone entries for these
# letters where one exists (D/T retroflex consonants and O keep their
# capital, matching this project's phoneme convention; the standalone সি
# entry was itself wrong -- "sh i" -- the user's example (সিআরপিএফ/CRPF)
# is what caught it; English letter "C" is plain "s i", not "sh i").
# Longer alternate spellings listed first so longest-match tokenizing works.
LETTERS = [
    ("ডাব্লিউ", "D a bl i u"),
    ("ডব্লিউ", "D a bl i u"),
    ("ওয়াই", "o a i"),
    ("এইচ", "e i ch"),
    ("কিউ", "k i u"),
    ("এক্স", "e k s"),
    ("জেড", "j e D"),
    ("এ", "e"),
    ("বি", "b i"),
    ("সি", "s i"),
    ("ডি", "D i"),
    ("ই", "i"),
    ("এফ", "e ph"),
    ("জি", "j i"),
    ("এচ", "e ch"),
    ("আই", "a i"),
    ("জে", "j e"),
    ("কে", "k e"),
    ("এল", "e l"),
    ("এম", "e m"),
    ("এন", "e n"),
    ("ও", "O"),
    ("পি", "p i"),
    ("আর", "a r"),
    ("এস", "e s"),
    ("টি", "T i"),
    ("ইউ", "i u"),
    ("ভি", "bh i"),
]
LETTERS.sort(key=lambda t: -len(t[0]))

# Real Bangla words (pronouns, particles, demonstratives, kinship terms,
# loanwords read as a whole word, not spelled out letter-by-letter) that
# happen to collide with a short letter-name sequence. Found by eyeballing
# every 2-token match in a dry run -- the risk zone for false positives,
# since true abbreviations as short as 2 letters (DC, MP, BA...) and real
# short Bangla words are both common at that length.
EXCLUDE = {
    "আরজি",   # petition/application (native)
    "ওই", "এই", "এটি", "এটিই", "এও",  # demonstratives
    "কেই", "ইকে", "একে", "ওকে",  # pronoun case forms
    "জেই",    # whoever/as soon as (normally যেই)
    "আরও",    # "more/also" -- very common native word
    "বিবি",   # wife/begum (Persian/Urdu loanword, nativized)
    "ওটি",    # "that one" demonstrative
    "আরবি", "আরবিতে", "আরবির",  # "Arabic" (the language), native word
    "ভিডিও",  # "video" read as one loanword, like রেডিও/স্টুডিও, not V-D-O
    "ইজি", "ইজির", "ইজিতে",  # borrowed "easy" (ইজি চেয়ার), not abbreviation E-G
    "পিওর",   # "pure" read as one loanword, not P.O.+possessive
    "পিসি",   # paternal aunt (kinship term)
    "সিটি",   # "city" read as one loanword, not spelled out C-I-T-Y
}

# Roots to block as a PREFIX, not just an exact word -- these loanwords take
# classifiers/case suffixes (ভিডিওটি "the video", পিওর "pure") that the
# suffix-stripping pass would otherwise peel back down to a colliding
# letter-name stem (ভিডিও -> strip টি -> still "video"; পিওর -> strip র ->
# পিও, which IS a real abbreviation (P.O.) in isolation, so the exact-word
# EXCLUDE above doesn't catch the suffixed form).
EXCLUDE_PREFIXES = ("ভিডিও", "পিওর", "ইজি", "আরবি")


def tokenize(word):
    """Greedy longest-match decomposition. Returns list of phonemes, or None
    if the word doesn't decompose exactly into >=2 letter tokens."""
    if word in EXCLUDE:
        return None
    i, toks = 0, []
    while i < len(word):
        for sp, ph in LETTERS:
            if word.startswith(sp, i):
                toks.append(ph)
                i += len(sp)
                break
        else:
            return None
    return toks if len(toks) >= 2 else None


# Bangla case/plural suffixes safe to strip before checking for a letter-name
# stem. "ই"/"ও"/"কে" (and কেই/কেও) are deliberately excluded: they're also
# literal letter names (I/O/K) and single-char emphasis/case particles, so
# stripping them is high collision risk for near-zero real matches.
SUFFIXES = ["দের", "এর", "ের", "সহ", "রা", "র", "তে", "হীন", "ভুক্ত"]
SUFFIXES.sort(key=len, reverse=True)

SUFFIX_PHON = {
    "দের": "d e r", "এর": "e r", "ের": "e r", "সহ": "sh o h O",
    "রা": "r a", "র": "r", "তে": "t e", "হীন": "h i n", "ভুক্ত": "bh u kt O",
}


def tokenize_with_suffix(word):
    """Like tokenize(), but also tries stripping one trailing case/plural
    suffix first. Returns (phoneme_tokens, suffix_phon_or_None) or None."""
    if word.startswith(EXCLUDE_PREFIXES):
        return None
    toks = tokenize(word)
    if toks is not None:
        return toks, None
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf):
            stem = word[: -len(suf)]
            toks = tokenize(stem)
            if toks is not None:
                return toks, SUFFIX_PHON[suf]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="../data/lexicon.tsv")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--suffixes", action="store_true",
                     help="also match abbreviation stem + case/plural suffix")
    args = ap.parse_args()

    with io.open(args.lexicon, encoding="utf-8") as f:
        lines = f.readlines()

    changes = []
    for i, line in enumerate(lines):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        word, phon, tag = parts
        if args.suffixes:
            result = tokenize_with_suffix(word)
            if result is None:
                continue
            toks, suf_phon = result
            new_phon = " ".join(toks) + (f" {suf_phon}" if suf_phon else "")
        else:
            toks = tokenize(word)
            if toks is None:
                continue
            new_phon = " ".join(toks)
        if (new_phon, tag) != (phon, "abbreviation"):
            changes.append((i, word, phon, tag, new_phon))

    print(f"{len(changes)} words recognized as letter-name abbreviations\n")
    for i, word, old_phon, old_tag, new_phon in changes:
        print(f"  {word}\told=({old_phon}|{old_tag})\tnew=({new_phon}|abbreviation)")

    if args.write:
        for i, word, old_phon, old_tag, new_phon in changes:
            lines[i] = f"{word}\t{new_phon}\tabbreviation\n"
        with io.open(args.lexicon, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"\nWrote {len(changes)} fixes to {args.lexicon}")
    else:
        print("\nDry run only -- rerun with --write to apply.")


if __name__ == "__main__":
    main()
