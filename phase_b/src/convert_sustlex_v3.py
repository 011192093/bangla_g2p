"""
SUSTlex converter v3 -- fixes the decomposed/composed Unicode bug for
য়/ড়/ঢ় using explicit \\uXXXX escapes (no ambiguity this time), plus
the earlier ্যা->A fix.
"""

# Explicit composed-form characters via escapes -- guaranteed correct,
# no reliance on how a literal character was typed/saved
YA_WITH_NUKTA_COMPOSED  = "\u09DF"  # য়
RA_WITH_NUKTA_COMPOSED  = "\u09DC"  # ড়
RHA_WITH_NUKTA_COMPOSED = "\u09DD"  # ঢ়

VOWELS = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u",
    "ঋ": "ri", "এ": "e", "ঐ": "oi", "ও": "O", "ঔ": "ou",
}
VOWEL_SIGNS = {
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u",
    "ৃ": "ri", "ে": "e", "ৈ": "oi", "ো": "O", "ৌ": "ou",
}
CONSONANTS = {
    "ক": "k",  "খ": "kh", "গ": "g",  "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "chh","জ": "j",  "ঝ": "jh", "ঞ": "ny",
    "ট": "T",  "ঠ": "Th", "ড": "D",  "ঢ": "Dh", "ণ": "n",
    "ত": "t",  "থ": "th", "দ": "d",  "ধ": "dh", "ন": "n",
    "প": "p",  "ফ": "ph", "ব": "b",  "ভ": "bh", "ম": "m",
    "য": "j",  "র": "r",  "ল": "l",
    "শ": "sh", "ষ": "sh", "স": "sh", "হ": "h",
    # explicit composed forms via escape -- guaranteed correct
    RA_WITH_NUKTA_COMPOSED:  "r",
    RHA_WITH_NUKTA_COMPOSED: "rh",
    YA_WITH_NUKTA_COMPOSED:  "y",
}
HASANTA = "্"
CHANDRABINDU = "ঁ"
ANUSVARA = "ং"
VISARGA = "ঃ"
NUKTA = "\u09BC"
YA_PHOLA = "য"

# Decomposed sequences -> TRUE composed forms (explicit escapes on both sides)
DECOMPOSED_MAP = {
    "\u09AF" + NUKTA: YA_WITH_NUKTA_COMPOSED,
    "\u09A1" + NUKTA: RA_WITH_NUKTA_COMPOSED,
    "\u09A2" + NUKTA: RHA_WITH_NUKTA_COMPOSED,
}

def preprocess(text):
    for decomposed, composed in DECOMPOSED_MAP.items():
        text = text.replace(decomposed, composed)
    return text


def convert_respelling(text):
    text = preprocess(text)
    phonemes = []
    had_unknown = False
    unknown_chars_found = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch in CONSONANTS:
            cluster = [ch]
            i += 1
            while i + 1 < n and text[i] == HASANTA and text[i+1] in CONSONANTS:
                cluster.append(text[i+1])
                i += 2

            explicit_no_vowel = False
            if i < n and text[i] == HASANTA:
                explicit_no_vowel = True
                i += 1

            vowel_sign = None
            nasalized = False
            if i < n and text[i] in VOWEL_SIGNS:
                vowel_sign = text[i]
                i += 1
            elif i < n and text[i] == CHANDRABINDU:
                nasalized = True
                i += 1

            if len(cluster) >= 2 and cluster[-1] == YA_PHOLA and vowel_sign == "া":
                for c in cluster[:-1]:
                    phonemes.append(CONSONANTS[c])
                phonemes.append("A")
                vowel_sign = None
            else:
                for c in cluster:
                    phonemes.append(CONSONANTS[c])
                if vowel_sign:
                    phonemes.append(VOWEL_SIGNS[vowel_sign])
                elif explicit_no_vowel:
                    pass
                else:
                    phonemes.append("o")

            if nasalized and phonemes:
                phonemes[-1] = phonemes[-1] + "~"

            if i < n and text[i] == ANUSVARA:
                phonemes.append("ng")
                i += 1
            elif i < n and text[i] == VISARGA:
                phonemes.append("h")
                i += 1

        elif ch in VOWELS:
            phonemes.append(VOWELS[ch])
            i += 1
            if i < n and text[i] == HASANTA:
                i += 1  # skip stray hasanta after a standalone vowel (glide marker, no phoneme needed)
            if i < n and text[i] == CHANDRABINDU:
                phonemes[-1] = phonemes[-1] + "~"
                i += 1
            if i < n and text[i] == ANUSVARA:
                phonemes.append("ng")
                i += 1
            elif i < n and text[i] == VISARGA:
                phonemes.append("h")
                i += 1

        elif ch.isspace() or ch in "–—-‌‍\ufeff":
            i += 1
        else:
            had_unknown = True
            unknown_chars_found.append(ch)
            i += 1

    return phonemes, had_unknown, unknown_chars_found


if __name__ == "__main__":
    tests = [
        ("অকুতোভয়্", ["o", "k", "u", "t", "O", "bh", "o", "y"]),
        ("ওই্সে", ["O", "i", "sh", "e"]),
        ("অক্", None),
    ]
    for respelling, expected in tests:
        result, unknown, chars = convert_respelling(respelling)
        status = "PASS" if expected and result == expected else "CHECK"
        print(f"{status}  {respelling} -> {result}  unknown={unknown} {chars if unknown else ''}")
