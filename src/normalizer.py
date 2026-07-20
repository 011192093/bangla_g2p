"""
Bangla Text Normalizer
======================
Run this BEFORE g2p_engine.py.

Pipeline:
  raw text
    -> convert Bangla digits to words
    -> expand common abbreviations
    -> handle English words mixed in (transliterate or skip)
    -> strip punctuation
    -> clean whitespace
    -> ready for text_to_phonemes()

HOW TO EXPAND THIS FILE:
  As you find new cases (abbreviations, number formats, etc.)
  just add them to the relevant dictionary or function below.
  Each function is small and independent — easy to test one at a time.
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# 1. BANGLA DIGIT → WORD
#    Each Bangla digit maps to its spoken Bangla word.
# ─────────────────────────────────────────────────────────────────────────────

ONES = {
    "0": "শূন্য",
    "1": "এক",
    "2": "দুই",
    "3": "তিন",
    "4": "চার",
    "5": "পাঁচ",
    "6": "ছয়",
    "7": "সাত",
    "8": "আট",
    "9": "নয়",
    "10": "দশ",
    "11": "এগারো",
    "12": "বারো",
    "13": "তেরো",
    "14": "চোদ্দো",
    "15": "পনেরো",
    "16": "ষোলো",
    "17": "সতেরো",
    "18": "আঠারো",
    "19": "উনিশ",
    "20": "বিশ",
    "21": "একুশ",       "22": "বাইশ",      "23": "তেইশ",       "24": "চব্বিশ",
    "25": "পঁচিশ",      "26": "ছাব্বিশ",   "27": "সাতাশ",      "28": "আটাশ",
    "29": "ঊনত্রিশ",    "30": "ত্রিশ",
    "31": "একত্রিশ",    "32": "বত্রিশ",    "33": "তেত্রিশ",     "34": "চৌত্রিশ",
    "35": "পঁয়ত্রিশ",   "36": "ছত্রিশ",    "37": "সাঁইত্রিশ",   "38": "আটত্রিশ",
    "39": "ঊনচল্লিশ",   "40": "চল্লিশ",
    "41": "একচল্লিশ",   "42": "বিয়াল্লিশ", "43": "তেতাল্লিশ",   "44": "চুয়াল্লিশ",
    "45": "পঁয়তাল্লিশ", "46": "ছেচল্লিশ",  "47": "সাতচল্লিশ",   "48": "আটচল্লিশ",
    "49": "ঊনপঞ্চাশ",   "50": "পঞ্চাশ",
    "51": "একান্ন",     "52": "বায়ান্ন",   "53": "তিপ্পান্ন",    "54": "চুয়ান্ন",
    "55": "পঞ্চান্ন",   "56": "ছাপ্পান্ন",  "57": "সাতান্ন",     "58": "আটান্ন",
    "59": "ঊনষাট",     "60": "ষাট",
    "61": "একষট্টি",    "62": "বাষট্টি",    "63": "তেষট্টি",      "64": "চৌষট্টি",
    "65": "পঁয়ষট্টি",   "66": "ছেষট্টি",    "67": "সাতষট্টি",     "68": "আটষট্টি",
    "69": "ঊনসত্তর",    "70": "সত্তর",
    "71": "একাত্তর",    "72": "বাহাত্তর",   "73": "তিয়াত্তর",    "74": "চুয়াত্তর",
    "75": "পঁচাত্তর",   "76": "ছিয়াত্তর",   "77": "সাতাত্তর",    "78": "আটাত্তর",
    "79": "ঊনআশি",     "80": "আশি",
    "81": "একাশি",     "82": "বিরাশি",     "83": "তিরাশি",      "84": "চুরাশি",
    "85": "পঁচাশি",     "86": "ছিয়াশি",    "87": "সাতাশি",      "88": "আটাশি",
    "89": "ঊননব্বই",    "90": "নব্বই",
    "91": "একানব্বই",   "92": "বিরানব্বই",  "93": "তিরানব্বই",    "94": "চুরানব্বই",
    "95": "পঁচানব্বই",  "96": "ছিয়ানব্বই",  "97": "সাতানব্বই",    "98": "আটানব্বই",
    "99": "নিরানব্বই",
    "1000": "এক হাজার",
}

# Bangla digit characters → Arabic digit characters
BANGLA_DIGIT_MAP = {
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
}


def bangla_digits_to_arabic(text: str) -> str:
    """
    Convert Bangla digit characters to Arabic numerals.
    e.g. "১২৩" -> "123"
    We do this first so the number-to-words function only
    needs to handle Arabic numerals.
    """
    result = text
    for bangla, arabic in BANGLA_DIGIT_MAP.items():
        result = result.replace(bangla, arabic)
    return result


def number_to_bangla_words(n: int) -> str:
    """
    Convert an integer to its spoken Bangla word form.
    Covers 0–9999 (extend for larger numbers as needed).

    Examples:
        5    -> "পাঁচ"
        23   -> "তেইশ"
        100  -> "একশো"
        1500 -> "এক হাজার পাঁচশো"
    """
    # Direct lookup for numbers we've hardcoded
    str_n = str(n)
    if str_n in ONES:
        return ONES[str_n]

    # Two-digit numbers (21–99) not already in ONES
    # Bangla has unique words for all numbers 1-99 (it's NOT simply "X-ty Y")
    # For now: return digits one by one as a fallback
    # TODO (Week 3 task): fill out the full 21–99 table in ONES above
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        tens_word = ONES.get(str(tens), "")
        ones_word = ONES.get(str(ones), "")
        return (tens_word + " " + ones_word).strip()

    if n < 1000:
        hundreds = n // 100
        remainder = n % 100
        # "-শো" (explicit vowel sign, not bare "শ"/"শত") always renders as a
        # clean "sho" regardless of what follows -- avoids relying on the
        # engine's inherent-vowel schwa rules, which get the boundary wrong
        # for a bound numeral suffix like this ("charsho", never "charosho"
        # or "charshoto")
        h_word = ONES.get(str(hundreds), str(hundreds)) + "শো"
        if remainder == 0:
            return h_word
        return h_word + " " + number_to_bangla_words(remainder)

    if n < 10000:
        thousands = n // 1000
        remainder = n % 1000
        t_word = ONES.get(str(thousands), str(thousands)) + " হাজার"
        if remainder == 0:
            return t_word
        return t_word + " " + number_to_bangla_words(remainder)

    # Fallback for very large numbers: return as-is (improve later)
    return str(n)


def normalize_numbers(text: str) -> str:
    """
    Find all runs of digits in the text and replace them with Bangla words.
    e.g. "আমার ১২ টাকা আছে" -> "আমার বারো টাকা আছে"
    """
    def replace_match(m):
        return number_to_bangla_words(int(m.group()))

    return re.sub(r"\d+", replace_match, text)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ABBREVIATION EXPANSION
#    Add entries here as you encounter them in your data.
# ─────────────────────────────────────────────────────────────────────────────

ABBREVIATIONS = {
    "ডঃ": "ডক্টর",
    "ড.": "ডক্টর",
    "মি.": "মিস্টার",
    "মিঃ": "মিস্টার",
    "প্রফেসর": "প্রফেসর",   # already full, no-op
    "বাং": "বাংলাদেশ",
    "ঢাঃ": "ঢাকা",
    "জা.বি": "জাহাঙ্গীরনগর বিশ্ববিদ্যালয়",
    "সু.বি": "সুপ্রিম কোর্ট",
    "কি.মি": "কিলোমিটার",
    "মি.মি": "মিলিমিটার",
    "সে.মি": "সেন্টিমিটার",
    "কেজি": "কিলোগ্রাম",
    # Latin-script abbreviations (must expand to Bangla script here, before
    # handle_english_words() strips remaining Latin letters)
    "BCS":   "বি সি এস",
    "PhD":   "পি এইচ ডি",
    "Engr.": "ইঞ্জিনিয়ার",
    "MBBS": "এম বি বি এস",
    "HSC": "এইচ এস সি",
    "SSC": "এস এস সি",
    "Mrs":   "মিসেস",
    "Ltd":   "লিমিটেড",
    "Govt":  "গভমেন্ট",
    "Dept":  "বিভাগ",
    "No":    "নম্বর",
    "Vol":   "ভলিউম",
    "Dr":    "ডক্টর",
    "Prof":  "প্রফেসর",
}


# \w alone doesn't cover Bengali vowel signs/virama/anusvara etc. (they aren't
# "alphanumeric" to Python's re), so a plain \w boundary sees a false word edge
# inside a larger word whenever an abbreviation happens to be flanked by one of
# those -- e.g. "প্যাকেজিং" contains "কেজি" preceded by "া" and followed by "ং",
# both non-\w, so (?<!\w)কেজি(?!\w) wrongly matched mid-word. Widen the boundary
# to the whole Bengali Unicode block so it only matches actual word edges.
_WORD_CHAR = r'[\wঀ-৿]'


def expand_abbreviations(text: str) -> str:
    """
    Replace known abbreviations with their full spoken forms.
    Processes longer abbreviations first so substrings don't interfere.
    """
    # Sort by length descending so longer matches are tried first
    for abbr, expansion in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        text = re.sub(r'(?<!' + _WORD_CHAR + r')' + re.escape(abbr) + r'(?!' + _WORD_CHAR + r')',
                       expansion, text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENGLISH WORD HANDLING (code-mixing)
#    Bangla text often contains English words written in Latin script.
#    Strategy for now: remove them (mark with a placeholder or skip).
#    Later (Phase B+): transliterate using a Bangla-English G2P bridge.
# ─────────────────────────────────────────────────────────────────────────────

def handle_english_words(text: str) -> str:
    """
    Detect Latin-script words inside Bangla text and either:
      - Remove them (current safe default)
      - Or mark them with a tag for later handling

    TODO: Replace this with actual transliteration in a later phase.
    """
    # Remove sequences of Latin letters (English words)
    # Keep Bangla, digits, and spaces
    result = re.sub(r"[a-zA-Z]+", "", text)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. PUNCTUATION + WHITESPACE CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

# Bangla uses the danda (।) as sentence terminal — treat like a period
PUNCTUATION_TO_STRIP = r"[।\.!?,;:\"'()\[\]{}\-–—/\\]"


def strip_punctuation(text: str) -> str:
    return re.sub(PUNCTUATION_TO_STRIP, " ", text)


def clean_whitespace(text: str) -> str:
    # Collapse multiple spaces into one, strip leading/trailing
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5. MASTER NORMALIZE FUNCTION
#    Call this on any raw Bangla text before passing to text_to_phonemes().
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """
    Full normalization pipeline.
    Input:  raw Bangla text (may contain Bangla digits, abbrs, English words)
    Output: clean Bangla text ready for the G2P engine
    """
    text = bangla_digits_to_arabic(text)   # ১২৩ -> 123
    text = normalize_numbers(text)          # 123 -> একশত তেইশ
    text = expand_abbreviations(text)       # ডঃ -> ডক্টর
    text = handle_english_words(text)       # remove Latin-script words
    text = strip_punctuation(text)          # remove ।!?, etc.
    text = clean_whitespace(text)           # tidy spaces
    return text


# ─────────────────────────────────────────────────────────────────────────────
# QUICK MANUAL TEST  (run: python src\normalizer.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        "ডঃ আহমেদ ১২ জানুয়ারি জন্মগ্রহণ করেন।",
        "বাংলাদেশের মোট জনসংখ্যা ১৭০ মিলিয়ন।",
        "আমার mobile number হলো ০১৭১২৩৪৫৬৭৮।",
        "তাপমাত্রা ৩৮ ডিগ্রি সেলসিয়াস।",
        "মি. করিম ঢাঃ থেকে এসেছেন।",
    ]

    for raw in test_cases:
        cleaned = normalize(raw)
        print(f"INPUT : {raw}")
        print(f"OUTPUT: {cleaned}")
        print()
