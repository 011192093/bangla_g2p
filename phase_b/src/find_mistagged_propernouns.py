"""
Heuristic (non-ML) detector for likely proper nouns (place names, mainly;
person-name surnames as a second, smaller pass) tagged something other
than proper_noun in lexicon.tsv.

Method, place names: Bangladeshi place names heavily reuse a small set of
suffixes (-পুর, -গঞ্জ, -নগর, -বাজার, -গাছা, -হাট, -গড়, -ডাঙ্গা, -খালি,
-চর, -কাঠি, -দ্বীপ...). A word ending in one of these, currently tagged
native (place names are essentially never tatsama/compound/abbreviation),
is a candidate -- but the same suffixes also end ordinary common nouns
(e.g. বাজার itself, "market"), so this is a lower-precision pass than the
loanword lists: every hit needs a human look, more so than usual.

Method, surnames: a short reference list of common Bangladeshi Hindu/
Muslim surname elements, same mechanism as the Arabic name-element pass.

Usage:
    python find_mistagged_propernouns.py > candidates.tsv
"""
import io
import sys
sys.path.insert(0, ".")
from find_mistagged_loanwords import strip_suffix, SKIP_TAGS

LEXICON = "../data/lexicon.tsv"

# Suffixes distinctive enough of Bangladeshi place names that a native-
# tagged word ending in one is worth a look. Excludes suffixes too
# common in ordinary vocabulary (e.g. বাড়ি "house", যুক্ত-type combines).
PLACE_SUFFIXES = [
    "গঞ্জ", "নগর", "গাছা", "ডাঙ্গা", "ডাঙা", "খালি", "কাঠি",
    "গড়", "হাট", "চর", "দ্বীপ", "বন্দর", "পল্লি", "পল্লী",
]

# Known common (non-place) words ending in one of the suffixes above --
# excluded so they don't get flagged every run. Found by eyeballing the
# first pass; add to this list as more turn up.
PLACE_FALSE_POSITIVES = {
    "বাজার", "মহাজন", "ভাঙ্গা", "রাঙা",
}

SURNAME_ELEMENTS = set("""
চৌধুরী চৌধুরি তালুকদার সরকার মজুমদার মন্ডল মণ্ডল বিশ্বাস ঘোষ রায়
সেন বোস দাস মুখার্জি মুখোপাধ্যায় ব্যানার্জি বন্দ্যোপাধ্যায় চ্যাটার্জি
চট্টোপাধ্যায় গাঙ্গুলি গঙ্গোপাধ্যায় ভট্টাচার্য লাহিড়ি সান্যাল পাল নাগ
দত্ত গুপ্ত মিত্র দে কর নন্দী ঘটক আচার্য সিংহ রাহা চাকলাদার প্রামানিক
খান মির্জা সিকদার শেখ মোল্লা মণ্ডল আকন্দ হাওলাদার মাতুব্বর গাজী
পাটোয়ারী ব্যাপারী মণ্ডল ফকির দরবেশ পন্ডিত বর্মন রাজবংশী বাগচী
""".split())


def main():
    with io.open(LEXICON, encoding="utf-8") as f:
        lines = f.readlines()

    place_hits, surname_hits = [], []
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        word, phon, tag = parts
        if tag in SKIP_TAGS or tag == "proper_noun":
            continue
        if word in PLACE_FALSE_POSITIVES:
            continue
        if tag == "native" and any(word.endswith(suf) and len(word) > len(suf) + 1
                                     for suf in PLACE_SUFFIXES):
            place_hits.append((word, phon, tag))
        stem = strip_suffix(word)
        if stem in SURNAME_ELEMENTS or word in SURNAME_ELEMENTS:
            surname_hits.append((word, phon, tag, stem))

    print(f"{len(place_hits)} place-name-suffix candidates (suggest -> proper_noun)\n")
    for word, phon, tag in place_hits:
        print(f"{word}\t{phon}\t{tag}")

    print(f"\n{len(surname_hits)} surname-element candidates (suggest -> proper_noun)\n")
    for word, phon, tag, stem in surname_hits:
        print(f"{word}\t{phon}\t{tag}\t(stem: {stem})")


if __name__ == "__main__":
    main()
