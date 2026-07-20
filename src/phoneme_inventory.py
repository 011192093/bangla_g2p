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
    "ৎ": "t",
    "ড়": "r",  "ড়": "r",
    "ঢ়": "r",  "ঢ়": "r",
    "য়": "y",  "য়": "y",
}
CONJUNCT_OVERRIDES = {
    "ক্ষ": ["kkh"],
    "জ্ঞ": ["g", "y"],
    "জ্য": ["j"],
    "চ্য": ["ch"],
    "গ্য": ["g", "g"],
    "ত্ত": ["tt"],
    "ন্ত": ["nt"],
    "ঞ্চ": ["nch"],
    "ঞ্জ": ["nj"],
    "ষ্ণ": ["shN"],
    "ক্ত": ["kt"],
    "স্ত": ["st"],
    "থ্য": ["t", "th"],
    "ন্দ": ["nd"],
    "ম্প": ["mp"],
    "ম্ব": ["mb"],
    "ঙ্গ": ["ngg"],
    "ত্ব": ["tw"],
    "দ্ব": ["dw"],
    "ন্ধ": ["ndh"],
    "স্ব": ["sh"],
    "শ্ব": ["sh"],
    "ন্ন": ["nn"],
    "ম্ম": ["mm"],
    "ষ্ট": ["shT"],
    "ষ্ঠ": ["shTh"],
    "স্থ": ["sth"],
    "ক্র": ["kr"],
    "প্র": ["pr"],
    "ব্র": ["br"],
    "ত্র": ["tr"],
    "গ্র": ["gr"],
    "শ্র": ["s", "r"],
    "শ্ল": ["s", "l"],
    "ক্ল": ["kl"],
    "ব্ল": ["bl"],
    "ন্দ্র": ["ndr"],
    "ন্ধ্য": ["n", "dh"],
    "স্ন": ["sn"],
    "ন্য": ["n", "n"],
    "ষ্ম": ["sh", "sh"],
    "দ্য": ["d", "d"],
    "জ্ব": ["j"],
    "ত্য": ["t", "t"],
    "র্ঘ্য": ["r", "gh"],
    "ব্য": ["b"],
    "শ্য": ["sh"],
    "হ্ব": ["b", "b"],
    "হ্র": ["r"],
    "হ্য": ["h"],
    "হ্ন": ["n", "n"],
    "হ্ম": ["m", "m"],
    "ক্ষ্ম": ["kkh"],
    "দ্ম": ["d", "d"],
    "ত্ত্ব": ["t", "t"],
    "ক্য": ["k", "k"],
    "খ্য": ["kh"],
    "ক্ষ্য": ["kkh"],
    "স্য": ["sh", "sh"],
    "স্থ্য": ["s", "th"],
    "ক্ষ্ণ": ["k", "h", "n"],
    "ল্য": ["l", "l"],
    "ট্য": ["T", "T"],
    "ন্ট": ["n", "T"],
    "চ্ছ": ["ch", "chh"],
    "ট্র": ["T", "r"],
    "দ্র": ["d", "r"],
    "ষ্ক": ["sh", "k"],
    "ত্ম": ["t","t"],
    "স্ত্র": ["s", "t", "r"],
     #র-medial (very common)
    "র্ক": ["r", "k"],    # তর্ক, বর্ক
    "র্ত": ["r", "t"],    # মূর্ত
    "র্ম": ["r", "m"],    # ধর্ম, কর্ম
    "র্ব": ["r", "b"],    # গর্ব, সর্ব
    "র্জ": ["r", "j"],    # অর্জন
    "র্শ": ["r", "sh"],   # স্পর্শ
    "র্ষ": ["r", "sh"],   # বর্ষ, নববর্ষ
    "র্থ": ["r", "th"],   # অর্থ
    "র্ধ": ["r", "dh"],   # বর্ধন
    "র্ভ": ["r", "bh"],   # গর্ভ
    "র্ট": ["r", "T"],    # পোর্ট, কোর্ট
    "র্ড": ["r", "D"],    # বোর্ড, কার্ড
    "র্ন": ["r", "n"],    # বর্ণ
    "র্প": ["r", "p"],    # দর্প
    "র্ফ": ["r", "ph"],   # স্কার্ফ
    "র্স": ["r","s"],

    # ল-medial (common)
    "ল্প": ["l", "p"],    # বিকল্প, প্রকল্প
    "ল্ট": ["l", "T"],    # রিজাল্ট, ভোল্ট
    "ল্ড": ["l", "D"],    # বিল্ড, ওয়ার্ল্ড
    "ল্ম": ["l", "m"],    # ফিল্ম
    "ল্ক": ["l", "k"],    # তালক

    # দ conjuncts (very common)
    "দ্ধ": ["d", "dh"],       # বুদ্ধ, শুদ্ধ, সিদ্ধ
    "দ্দ": ["d", "d"],        # উদ্দেশ্য
    "দ্ভ": ["d", "bh"],   # অদ্ভুত
    "ধ্ব": ["dh", "w"],   # ধ্বনি, ধ্বংস
    "ধ্য": ["dh", "y"],   # সাধ্য
    "ধ্র": ["dh", "r"],   # ধ্রুব

    # ন conjuncts (common)
    "ন্ম": ["n", "m"],    # জন্ম
    "ন্ড": ["n", "D"],    # পান্ডা, ব্যান্ড
    "ন্ত্র": ["n", "t", "r"], # মন্ত্র, যন্ত্র
    "ণ্ট": ["n", "T"],    # ঘণ্টা
    "ণ্ঠ": ["n", "Th"],   # কণ্ঠ
    "ণ্ড": ["n", "D"],    # মণ্ডল, দণ্ড
    "ন্ব":["n","n"],

    # স conjuncts (common)
    "স্ক": ["s", "k"],    # স্কুল, স্কার্ফ
    "স্ট": ["s", "T"],    # স্টেশন
    "স্প": ["s", "p"],    # স্পষ্ট
    "স্ম": ["s", "m"],    # স্মৃতি
    "স্র": ["s", "r"],
    "সৃ": ["s","r"],    # স্রোত
    "স্ল":["s","l"],
    "স্ব": ["sh"],

    # প conjuncts
    "প্ত": ["p", "t"],    # গুপ্ত, লুপ্ত
    "প্ন": ["p", "n"],    # স্বপ্ন

    # ব conjuncts
    "ব্দ": ["b", "d"],    # শব্দ
    "ব্ধ": ["b", "dh"],   # লব্ধ
    "ভ্র": ["bh", "r"],   # ভ্রমণ

    # গ conjuncts
    "গ্ন": ["g", "n"],    # মগ্ন, লগ্ন
    "গ্ধ": ["g", "dh"],   # মুগ্ধ

    # ম conjuncts
    "ম্ভ": ["m", "bh"],   # সম্ভব

    # ট conjuncts
    "ট্ট": ["T", "T"],    # চট্টগ্রাম

    # ত conjuncts
    "ত্থ": ["t", "th"],   # উত্থান
    "ত্ন": ["t", "n"],    # যত্ন, রত্ন

    # শ conjuncts
    "শ্ন": ["sh", "n"],   # প্রশ্ন
    "শ্চ": ["sh", "ch"],  # নিশ্চয়

    # ফ conjuncts (loanwords)
    "ফ্র": ["ph", "r"],   # ফ্রি, ফ্রান্স
    "ফ্ল": ["ph", "l"],   # ফ্লাইট
    "ফ্ট": ["ph", "T"],   # লিফট, শিফট

    # Geminate conjuncts
    "ক্ক": ["k", "k"],    # মক্কা
    "চ্চ": ["ch", "ch"],  # বচ্চন
    "জ্জ": ["j", "j"],    # হজ্জ
    "ড্ড": ["D", "D"],

    # ষ conjuncts
    "ষ্প": ["sh", "p"],
    "চ্ছ্ব":["ch","chh"],
    "ষ্য": ["sh", "sh"],  # বিশেষ্য -- matches স্য's geminate pattern

    # য-medial (ya-phala family, glide silent)
    "প্য": ["p"],  # প্যাথলজিস্ট -- matches ব্য/শ্য/চ্য word-initial single pattern
    "ন্দ্ব": ["n", "d"],  # প্রতিদ্বন্দ্বী -- trailing ব silent, matches হ্ব/হ্ন-style drops
    "ণ্য": ["n", "n"],  # পণ্য -- ণ merges with ন, same geminate as ন্য
    "ন্ন্য": ["n", "n"],  # সন্ন্যাসী -- glide silent, same as ন্য/ণ্য
    "স্ট্য": ["s", "T"],  # স্ট্যাম্প -- glide silent, matches স্ট্যান্ড
}

# ব্য/শ্য-family clusters (ya-phala): the "য" is never pronounced as "j",
# and when no explicit vowel sign follows, the inherent vowel is "a" (not
# the usual "o" schwa) and is never suppressed by neighboring context.
YA_PHALA_INHERENT_A = {"ব্য", "শ্য"}

# The ে (e-kar) rule: only the WORD-INITIAL syllable's ে can ever be the
# open "ae" sound (phoneme "E") -- every later syllable's ে is always the
# closed "e" sound, no matter what follows. Within that word-initial
# syllable, ae happens in exactly two cases:
#   1. The next consonant carries া or ো, AND that consonant is the
#      word's LAST syllable (দেখা "dEkha", তেরো "tEro"). Position decides
#      here, not vowel quality -- কেরোসিন "kerosin" has the same ে+cons+ো
#      shape as তেরো but stays closed because রো isn't last there, and
#      চেয়ার stays closed for the same reason (্‌যা isn't last, র follows).
#   2. The next consonant carries no explicit vowel sign (inherent o) and
#      is a bare SONORANT, AND that sonorant is not itself word-final
#      (কেমন "kEmon", but তেল "tel" closes because ল ends the word there).
# Everything else -- before ি/ী/ু/ূ/ে, before a য় glide, before a
# non-final া/ো, before an inherent-vowel OBSTRUENT, before a word-final
# bare consonant of any kind, or simply not word-initial -- is plain e.
SONORANT_CONSONANTS = {"ঙ", "ঞ", "ণ", "ন", "ম", "ল", "র"}
HASANTA = "্"
CHANDRABINDU = "ঁ"
ANUSVARA = "ং"
VISARGA = "ঃ"
DIACRITIC_PHONEMES = {"ং": "ng", "ঃ": "h"}
BANGLA_DIGITS = {
    "০": "0","১": "1","২": "2","৩": "3","৪": "4",
    "৫": "5","৬": "6","৭": "7","৮": "8","৯": "9",
}
ALL_VOWEL_CHARS = set(VOWELS.keys())
ALL_VOWEL_SIGN_CHARS = set(VOWEL_SIGNS.keys())
ALL_CONSONANT_CHARS = set(CONSONANTS.keys())

def classify_char(ch):
    if ch in ALL_VOWEL_CHARS: return "VOWEL"
    if ch in ALL_VOWEL_SIGN_CHARS: return "VOWEL_SIGN"
    if ch in ALL_CONSONANT_CHARS: return "CONSONANT"
    if ch == HASANTA: return "HASANTA"
    if ch == CHANDRABINDU: return "CHANDRABINDU"
    if ch in DIACRITIC_PHONEMES: return "DIACRITIC"
    if ch in BANGLA_DIGITS: return "DIGIT"
    if ch.isspace(): return "SPACE"
    return "OTHER"
