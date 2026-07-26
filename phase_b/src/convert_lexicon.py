"""
Phase B — Lexicon Converter
============================
Converts the Phase A rule engine's lexicon_seed.tsv into the format
Phase B's data pipeline expects: word<TAB>phonemes<TAB>register_tag.

lexicon_seed.tsv itself carries no register metadata (Phase A doesn't need
it) and stays untouched -- this script classifies register tags on the fly
for every entry that doesn't already have one, using signals available at
convert time:

  - REGISTER_OVERRIDES: hand-verified per-word corrections, checked before
                 anything else. Sourced from cross-checking every "native"
                 word against Bangla Academy's dictionary root-language data
                 (get_root_lang()) and a human review pass over the result:
                 122 Persian/Arabic loans old enough to be fully nativized
                 (গরম, চেয়ার, দরজা...) -> loanword_nativized; 7 recent loans
                 that still sound foreign (জেব্রা, শেয়ার, ক্যামেরা,
                 ফটোগ্রাফার, লুডু, ফ্ল্যাট, এজেন্ট) -> loanword; 4 the
                 dictionary data mis-flagged (ঢোল, বঁটি, বাংলা, জানা) confirmed
                 back to native; এস (a spelled-out Latin letter, not real
                 vocabulary) -> unknown, so the loanword signal doesn't
                 wrongly grab it. See phase_b/README or ask for the audit
                 artifact this was reviewed from if regenerating.
  - proper_noun: word matches a hand-curated list of place/country names,
                 days, Bangla calendar months, and person/religious names
                 (e.g. ঢাকা, বাংলাদেশ, সোমবার, বৈশাখ, আল্লাহ). Unlike the other
                 tags this can't be derived from spelling or phonemes -- it
                 needs word-level knowledge -- so it's a fixed list, not a
                 rule, and only covers what's actually in the lexicon today.
                 সিলেট, for instance, would otherwise get caught by the
                 loanword signal below (its "s" sound), but it's
                 fundamentally a place name.
  - compound   : phonemes contain "|" (a multi-word phrase entry)
  - loanword   : the lexicon overrides স's default "sh" to "s" -- either via
                 a conjunct pattern calibrated for English loanwords
                 (স্ক/স্ট/স্প/স্ম/স্ট্য, e.g. স্কুল, স্টেশন), or a hand-verified
                 per-word override where the raw rule engine still says "sh"
                 (e.g. সাইকেল, সিনেমা). NOT triggered by conjuncts that render
                 literal "s" as a native Sanskrit-pattern artifact regardless
                 of register (শ্র, শ্ল, স্থ্য, স্ত্র, র্স, স্র, সৃ, স্ল --
                 শ্রাবণ, স্ত্রী, স্রোত are native/tatsama, not loanwords).
  - tatsama    : spelling contains ণ, ষ, ৃ, or ঞ -- these are essentially
                 exclusive to Sanskrit-derived words in Bangla orthography.
  - native     : default fallback (also covers tadbhava/deshi, which aren't
                 reliably distinguishable from spelling alone).

This is a best-effort heuristic, not authoritative etymology -- known
false positives include tatsama words where স realizes as "s" for a
Sanskrit-internal phonetic reason rather than foreign borrowing (e.g.
সৃষ্টি). Spot-check before trusting it for anything beyond giving the
model a training signal.

Usage:
    python convert_lexicon.py --input ..\\..\\data\\lexicon_seed.tsv --output ..\\data\\lexicon.tsv
"""

import os
import sys
import argparse
import difflib
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import g2p_engine as G
from normalizer import normalize

VALID_TAGS = {
    "unknown", "native", "tatsama", "tadbhava", "deshi",
    "loanword", "loanword_nativized", "compound", "proper_noun",
}

LOANWORD_CONJUNCTS = {"স্ক", "স্ট", "স্প", "স্ম", "স্ট্য"}

# Hand-curated -- everything currently in lexicon_seed.tsv that's a named
# entity rather than a common word. Extend this list by hand as the lexicon
# grows; there's no reliable way to detect a proper noun from spelling alone.
PROPER_NOUNS = {
    # Bangladeshi divisional cities
    "ঢাকা", "চট্টগ্রাম", "রাজশাহী", "খুলনা", "সিলেট", "বরিশাল",
    "ময়মনসিংহ", "রংপুর", "কুমিল্লা", "নারায়ণগঞ্জ",
    # Countries and continents
    "বাংলাদেশ", "ভারত", "পাকিস্তান", "নেপাল", "চীন", "ইউরোপ", "আফ্রিকা",
    # Days of the week
    "রবিবার", "সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার",
    # Bangla calendar months
    "বৈশাখ", "জ্যৈষ্ঠ", "আষাঢ়", "শ্রাবণ", "ভাদ্র", "আশ্বিন",
    "কার্তিক", "অগ্রহায়ণ", "পৌষ", "মাঘ", "ফাল্গুন", "চৈত্র",
    # Person and religious names
    "সেলিম", "সাকলাইন", "আল্লাহ",
}

# Hand-verified against Bangla Academy's dictionary root-language data,
# then reviewed by hand -- see the note above. Highest priority: checked
# before PROPER_NOUNS and the structural heuristics below.
REGISTER_OVERRIDES = {}
REGISTER_OVERRIDES.update({w: "loanword_nativized" for w in [
    "খারাপ", "তরকারি", "সবুজ", "গরম", "মুরগি", "হাজার", "খুশি", "উকিল",
    "দর্জি", "দরজা", "জানালা", "জামা", "খালা", "তবলা", "আঙ্গুর", "তরমুজ",
    "কাগজ", "মসজিদ", "নামাজ", "ঈদ", "নরম", "বালিশ", "পর্দা", "ওজন",
    "পোশাক", "মসলা", "চাকরি", "সরকার", "আইন", "আদালত", "নগদ", "খামার",
    "সাবান", "ইলিশ", "জাহাজ", "বাকি", "খুচরা", "দোকানদার", "সরবরাহ",
    "তোশক", "মোজা", "চশমা", "জামিন", "মোমবাতি", "বারান্দা", "কাদের",
    "আফসোস", "সুদ", "মুনাফা", "লোকসান", "সিকি", "কমবেশি", "মালিক",
    "ইমাম", "চাদর", "আসলে", "রেশম", "পশম", "পনির", "শরবত", "মামলা",
    "জরিমানা", "গ্রেপ্তার", "মোছা", "গোলাপ", "বীমা", "রশিদ", "নোঙর",
    "কারখানা", "জায়নামাজ", "মেরামত", "মানে", "সেতার", "শাল", "কবরস্থান",
    "বন্দর", "মলম", "তাস", "কুস্তি", "দাফন", "রপ্তানি", "আমদানি",
    "দালান", "মেহেদি", "দলিল",
    # from the "loanword" bucket, reclassified nativized on review
    "পুলিশ", "চেয়ার", "টেবিল", "আলমারি", "ট্রেন", "রিকশা", "চাচা",
    "ফুফু", "ফুফা", "পেঁপে", "লিচু", "গির্জা", "বাটি", "টাকা", "পেরেক",
    "আচ্ছা", "অটোরিকশা", "বোতল", "চিনি", "চাবি", "চিঠি", "রাজমিস্ত্রি",
    "ইস্ত্রি", "জিলাপি", "ইংরেজ", "ডাকাত", "টব", "গিরগিটি", "মাস্তুল",
    "আনারস", "রেললাইন", "ক্যারাম", "বোতাম", "চিড়িয়াখানা", "লেনদেন",
    "চাটনি", "গুদাম",
]})
REGISTER_OVERRIDES.update({w: "loanword" for w in [
    "জেব্রা", "শেয়ার", "ক্যামেরা", "ফটোগ্রাফার", "লুডু", "ফ্ল্যাট", "এজেন্ট",
]})
REGISTER_OVERRIDES.update({w: "native" for w in [
    "ঢোল", "বঁটি", "বাংলা", "জানা",
]})
# Spelled-out Latin letter names (from MBBS-style abbreviation contexts) --
# a closed, small set the lexicon lookup already handles perfectly on its
# own, no register signal needed. Overridden to "unknown" only to keep it
# out of "loanword", which the structural heuristic below would otherwise
# assign (raw engine says "sh", lexicon says "s" -- same shape as a real
# loanword override, just coincidentally).
REGISTER_OVERRIDES.update({w: "unknown" for w in [
    "এস",
]})


def word_has_loanword_s(word: str, lex_phon_toks: list) -> bool:
    if "|" in word:
        return False
    clusters = G.segment_clusters(normalize(word))
    for i, c in enumerate(clusters):
        if getattr(c, "_passthrough", None) is not None or not c.consonants:
            continue
        cstr = G.cluster_to_orthographic_string(c.consonants)
        if cstr in LOANWORD_CONJUNCTS:
            return True
    try:
        raw = G.text_to_phonemes(normalize(word))
    except Exception:
        return False
    sm = difflib.SequenceMatcher(None, raw, lex_phon_toks, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            a, b = raw[i1:i2], lex_phon_toks[j1:j2]
            if "sh" in a and "s" in b:
                return True
    return False


def classify_register(word: str, phonemes: str) -> str:
    if word in REGISTER_OVERRIDES:
        return REGISTER_OVERRIDES[word]
    if word in PROPER_NOUNS:
        return "proper_noun"
    if "|" in phonemes:
        return "compound"
    toks = phonemes.split()
    if "s" in toks and word_has_loanword_s(word, toks):
        return "loanword"
    if any(ch in word for ch in "ণষৃঞ"):
        return "tatsama"
    return "native"


def convert(input_path: str, output_path: str) -> None:
    entries = []
    tag_counts = Counter()
    auto_tagged = 0
    skipped = 0

    with open(input_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                skipped += 1
                continue
            word = parts[0].strip()
            phonemes = parts[1].strip()
            if not word or not phonemes:
                skipped += 1
                continue
            tag = parts[2].strip() if len(parts) >= 3 else ""
            if tag and tag not in VALID_TAGS:
                print(f"  line {line_no}: unrecognized tag {tag!r} for {word!r} -- writing it anyway, data.py will fall back to 'unknown'")
            elif not tag:
                tag = classify_register(word, phonemes)
                auto_tagged += 1
            tag_counts[tag] += 1
            entries.append((word, phonemes, tag))

    with open(output_path, "w", encoding="utf-8") as f:
        for word, phonemes, tag in entries:
            f.write(f"{word}\t{phonemes}\t{tag}\n")

    print(f"Converted {len(entries)} entries -> {output_path}")
    print(f"  {auto_tagged} were auto-classified (no tag in lexicon_seed.tsv); {len(entries) - auto_tagged} already had one")
    if skipped:
        print(f"Skipped {skipped} malformed line(s)")
    print("Register tag breakdown:")
    for tag, n in tag_counts.most_common():
        print(f"  {n:5d}  {tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert lexicon_seed.tsv to Phase B's lexicon.tsv format")
    parser.add_argument("--input", required=True, help="Path to lexicon_seed.tsv")
    parser.add_argument("--output", required=True, help="Path to write lexicon.tsv")
    args = parser.parse_args()
    convert(args.input, args.output)
