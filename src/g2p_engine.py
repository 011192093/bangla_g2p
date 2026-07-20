"""
Bangla Rule-Based G2P Engine
"""

import unicodedata
from typing import List
from phoneme_inventory import (
    VOWELS, VOWEL_SIGNS, CONSONANTS, CONJUNCT_OVERRIDES,
    HASANTA, CHANDRABINDU, DIACRITIC_PHONEMES, classify_char,
    YA_PHALA_INHERENT_A, SONORANT_CONSONANTS,
)

KHANDA_TA = "\u09CE"

# Decomposed sequences that NFC does not automatically combine.
# Replace these before any processing.
NUKTA = "\u09BC"
DECOMPOSED_MAP = {
    "\u09AF" + NUKTA: "\u09DF",   # য + ় → য় (yya)
    "\u09A1" + NUKTA: "\u09DC",   # ড + ় → ড় (rra)
    "\u09A2" + NUKTA: "\u09DD",   # ঢ + ় → ঢ় (rha)
}


def preprocess(text: str) -> str:
    """NFC normalize then fix decomposed nukta sequences."""
    text = unicodedata.normalize("NFC", text)
    for decomposed, composed in DECOMPOSED_MAP.items():
        text = text.replace(decomposed, composed)
    return text


class Cluster:
    def __init__(self):
        self.consonants: List[str] = []
        self.vowel_sign: str = None
        self.standalone_vowel: str = None
        self.nasalized: bool = False
        self.trailing_diacritic: str = None


def segment_clusters(text: str) -> List[Cluster]:
    clusters: List[Cluster] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        cls = classify_char(ch)
        if cls == "CONSONANT":
            c = Cluster()
            c.consonants.append(ch)
            i += 1
            while i + 1 < n and text[i] == HASANTA and classify_char(text[i + 1]) == "CONSONANT":
                c.consonants.append(text[i + 1])
                i += 2
            explicit_no_vowel = False
            if i < n and text[i] == HASANTA:
                explicit_no_vowel = True
                i += 1
            if i < n and classify_char(text[i]) == "VOWEL_SIGN":
                c.vowel_sign = text[i]
                i += 1
            elif explicit_no_vowel:
                c.vowel_sign = "__NONE__"
            if i < n and classify_char(text[i]) == "CHANDRABINDU":
                c.nasalized = True
                i += 1
            if i < n and text[i] in DIACRITIC_PHONEMES:
                c.trailing_diacritic = text[i]
                i += 1
            clusters.append(c)
        elif cls == "VOWEL":
            c = Cluster()
            c.standalone_vowel = ch
            i += 1
            if i < n and classify_char(text[i]) == "CHANDRABINDU":
                c.nasalized = True
                i += 1
            if i < n and text[i] in DIACRITIC_PHONEMES:
                c.trailing_diacritic = text[i]
                i += 1
            clusters.append(c)
        else:
            c = Cluster()
            c.standalone_vowel = None
            c.consonants = []
            c._passthrough = ch
            i += 1
            clusters.append(c)
    return clusters


def cluster_to_orthographic_string(cluster_chars: List[str]) -> str:
    return HASANTA.join(cluster_chars)


def is_word_final(clusters: List[Cluster], idx: int) -> bool:
    for j in range(idx + 1, len(clusters)):
        c = clusters[j]
        if getattr(c, "_passthrough", None) is not None:
            return c._passthrough.isspace() or not c._passthrough.isalnum()
        return False
    return True


def _is_word_initial(clusters: List[Cluster], idx: int) -> bool:
    if idx == 0:
        return True
    prev = clusters[idx - 1]
    if getattr(prev, "_passthrough", None) is not None:
        return prev._passthrough.isspace() or not prev._passthrough.isalnum()
    return False


HIGH_FRONT_VOWEL_SIGNS = {"ি", "ী", "ু", "ূ", "ে"}
HIGH_FRONT_STANDALONE_VOWELS = {"ই", "ঈ", "উ", "ঊ", "এ"}


def _next_syllable_triggers_harmony(clusters: List[Cluster], idx: int) -> bool:
    # Bengali vowel height harmony: a bare consonant's inherent vowel
    # (normally open, "aw" as in মহা "mawha") raises to the closed "O"
    # sound when the *next* syllable has an explicit high/front vowel
    # (ি/ী/ু/ূ/ে, as in মণি "moni"). A following inherent/schwa vowel, or
    # an open vowel sign (া etc.), does not trigger raising.
    for j in range(idx + 1, len(clusters)):
        next_c = clusters[j]
        if getattr(next_c, "_passthrough", None) is not None:
            return False
        if next_c.standalone_vowel is not None:
            return next_c.standalone_vowel in HIGH_FRONT_STANDALONE_VOWELS
        return next_c.vowel_sign in HIGH_FRONT_VOWEL_SIGNS
    return False


PENULTIMATE_HARMONY_SONORANTS = {"ম", "ন", "ণ", "ল", "র", "ড়"}


def _word_syllable_count(clusters: List[Cluster], idx: int) -> int:
    start = idx
    while not _is_word_initial(clusters, start):
        start -= 1
    end = idx
    n = len(clusters)
    while end + 1 < n and not is_word_final(clusters, end):
        end += 1
    return end - start + 1


def _penultimate_sonorant_triggers(clusters: List[Cluster], idx: int) -> bool:
    # A second, independent raising trigger (distinct from the high-vowel
    # harmony above): in a word of 3+ syllables, a bare consonant raises
    # to "O" when it's immediately followed by a word-final SONORANT
    # (ম/ন/ণ/ল/র/ড়) that itself carries only its inherent vowel, AND the
    # PRECEDING syllable is itself a bare single consonant carrying either
    # its own inherent vowel or া -- গরম "gorOm", কলম "kolOm", লবণ
    # "lobOn", ছাগল "chhagOl". A conjunct preceding syllable (শ্রাবণ,
    # বিশ্লেষণ) or one with a ে vowel sign (কেমন) does NOT trigger this,
    # even though ে isn't itself a "close" vowel -- both stay unraised.
    # Contrast সুন্দর, where the raising consonant is a conjunct (ন্দ)
    # rather than bare -- that stays a lexicon-only exception, same as
    # this rule's conjunct-exempt sibling above.
    if _word_syllable_count(clusters, idx) < 3:
        return False
    next_j = _next_real_cluster_idx(clusters, idx)
    if next_j is None or not is_word_final(clusters, next_j):
        return False
    next_c = clusters[next_j]
    if next_c.vowel_sign is not None and next_c.vowel_sign != "__NONE__":
        return False  # next consonant must carry only its inherent vowel
    if not (next_c.consonants and next_c.consonants[-1] in PENULTIMATE_HARMONY_SONORANTS):
        return False
    prev_c = clusters[idx - 1]
    if len(prev_c.consonants) != 1:
        return False  # preceding syllable must be a bare single consonant
    if prev_c.vowel_sign not in (None, "__NONE__", "া"):
        return False  # preceding vowel must be inherent or া, not ি/ী/ু/ূ/ে/etc.
    return True


def _next_cluster_is_consonant(clusters: List[Cluster], idx: int) -> bool:
    if _is_word_initial(clusters, idx):
        return False
    for j in range(idx + 1, len(clusters)):
        next_c = clusters[j]
        if getattr(next_c, "_passthrough", None) is not None:
            return False
        if (next_c.consonants
                and next_c.vowel_sign is not None
                and next_c.vowel_sign != "__NONE__"):
            return True
        return False
    return False


# --- Schwa deletion algorithm ---
# Decides whether a bare consonant cluster's inherent vowel (অ) is
# present at all. This replaces the old _next_cluster_is_consonant
# heuristic, which only correctly predicted a minority of confirmed
# cases this project has verified by hand. Rules are checked in order;
# the first one that applies wins. Validated against the full lexicon:
# ~81% of words now get the correct answer straight from the engine,
# up from ~66% under the old heuristic.


def _next_real_cluster(clusters: List[Cluster], idx: int):
    for j in range(idx + 1, len(clusters)):
        next_c = clusters[j]
        if getattr(next_c, "_passthrough", None) is not None:
            return None
        return next_c
    return None


def _next_real_cluster_idx(clusters: List[Cluster], idx: int):
    for j in range(idx + 1, len(clusters)):
        next_c = clusters[j]
        if getattr(next_c, "_passthrough", None) is not None:
            return None
        return j
    return None


def _eekar_is_open(clusters: List[Cluster], idx: int) -> bool:
    # ে (e-kar): open "ae" (phoneme "E") in exactly two conditions, and
    # only for the WORD-INITIAL syllable's ে -- every later syllable's ে
    # is always closed "e", no matter what follows (kEmon, khEla are
    # word-initial; hobe's ে is the second syllable and stays closed even
    # though hobe also happens to end the word). See SONORANT_CONSONANTS
    # in phoneme_inventory.py for the full rule writeup.
    if not _is_word_initial(clusters, idx):
        return False  # second+ syllable -> always e

    next_j = _next_real_cluster_idx(clusters, idx)
    if next_j is None:
        return False  # word-final -> e

    next_c = clusters[next_j]
    if next_c.standalone_vowel is not None:
        return False  # standalone vowel follows directly (কেউ) -> e

    if next_c.vowel_sign in ("া", "ো") and is_word_final(clusters, next_j):
        return True  # before া/ো on the word's LAST syllable -> ae (তেরো
        # "tEro", but কেরোসিন "kerosin" stays closed -- রো isn't last
        # there. ো doesn't trigger by itself; being the last syllable does
        # -- া works the same way, not unconditionally (মেশানো's া isn't
        # the last syllable, so it stays closed too).

    next_has_vowel = (next_c.vowel_sign is not None
                       and next_c.vowel_sign != "__NONE__")
    if next_has_vowel:
        return False  # ি/ী/ু/ূ/ে always closed; non-final া/ো also closed -> e

    # Next consonant carries only its inherent vowel (no explicit sign).
    # If THAT consonant itself ends the word (তেল "tel"), close regardless
    # of sonority -- ae only happens when a sonorant carries inherent o
    # AND the word still has more coming after it (কেমন "kEmon").
    if is_word_final(clusters, next_j):
        return False  # word-final -> e
    if next_c.consonants and next_c.consonants[-1] in SONORANT_CONSONANTS:
        return True  # sonorant with inherent o, not word-final -> ae
    return False  # obstruent (or য় glide) with inherent o -> e


def _schwa_should_keep(clusters: List[Cluster], idx: int) -> bool:
    cluster = clusters[idx]

    # Rule 1 — word-initial: always keep.
    if _is_word_initial(clusters, idx):
        return True

    # Rule 2 — word-final: single consonant drops, a conjunct keeps.
    if is_word_final(clusters, idx):
        return len(cluster.consonants) > 1

    # Rule 3 — look at the next cluster.
    next_c = _next_real_cluster(clusters, idx)
    if next_c is None:
        return True  # shouldn't happen (word-final already handled), default keep

    next_is_conjunct = len(next_c.consonants) > 1
    next_has_vowel = (next_c.vowel_sign is not None
                       and next_c.vowel_sign != "__NONE__")

    # 3a/3d — next cluster carries its own vowel sign (short or long) ->
    # drop, whether that cluster is a single consonant (আমরা "amra",
    # করছি "korchhi") or a conjunct that itself has a vowel (সাফল্যের's
    # ল্য carries ে, so ফ still drops -- "shapholler", not "shaphOller").
    if next_has_vowel:
        return False

    # 3b/3c — next cluster is a vowel-less conjunct (2+ consonants) ->
    # keep (e.g. রক্ত "rokto", চট্টগ্রাম "...T-T-o-gram").
    if next_is_conjunct:
        return True

    # 3e — next cluster is a bare single consonant with no vowel yet ->
    # drop, UNLESS that bare consonant is itself word-final (a bare
    # consonant at the very end of a word needs a linking vowel before
    # it: শ্যালক "shalok", লবণ "lobon", উৎসব "utshob").
    return is_word_final(clusters, idx + 1)


def cluster_to_phonemes(cluster: Cluster, clusters: List[Cluster], idx: int) -> List[str]:
    phonemes: List[str] = []

    if getattr(cluster, "_passthrough", None) is not None:
        return []

    if cluster.standalone_vowel is not None:
        next_idx = idx + 1
        if (next_idx < len(clusters)
                and getattr(clusters[next_idx], "_passthrough", None) == "্"):
            next_idx += 1  # hasanta joins the vowel to the next consonant
        next_c = clusters[next_idx] if next_idx < len(clusters) else None
        if (_is_word_initial(clusters, idx) and next_c is not None
                and getattr(next_c, "_passthrough", None) is None
                and next_c.consonants == ["য"] and next_c.vowel_sign == "া"):
            # অ্যা (vowel-initial ya-phala + aa-kar): the whole unit is a
            # single "A" vowel produced by the য cluster below -- this
            # vowel's own sound is absorbed into it (অ্যালবাম "Albam", not
            # "oJalbam")
            return []
        ph = VOWELS[cluster.standalone_vowel]
        phonemes.append(ph)
        if cluster.nasalized:
            phonemes[-1] = phonemes[-1] + "~"
        if cluster.trailing_diacritic:
            phonemes.append(DIACRITIC_PHONEMES[cluster.trailing_diacritic])
        return phonemes

    prev_idx = idx - 1
    if prev_idx >= 0 and getattr(clusters[prev_idx], "_passthrough", None) == "্":
        prev_idx -= 1  # hasanta joins the previous vowel to this consonant
    prev_c = clusters[prev_idx] if prev_idx >= 0 else None
    vowel_initial_ya_phala = (
        cluster.consonants == ["য"] and cluster.vowel_sign == "া"
        and prev_c is not None and prev_c.standalone_vowel is not None
        and _is_word_initial(clusters, prev_idx)
    )

    consonant_str = cluster_to_orthographic_string(cluster.consonants)
    if vowel_initial_ya_phala:
        phonemes.append("A")
    elif consonant_str in CONJUNCT_OVERRIDES:
        override = CONJUNCT_OVERRIDES[consonant_str]
        # জ্ঞ, শ্ব, জ্য, and হ্য all geminate mid-word/word-final (biggan,
        # bishsho, banijjo, grajjo) but stay single word-initial (gyan,
        # shoshur, joti/joishTho/jænto, hæ) -- gemination needs a
        # preceding vowel to double against. হ্য geminates as an aspirated
        # pair (j, jh) for সহ্য/অসহ্য specifically -- other হ্য words
        # (গ্রাহ্য, গ্রাহ্যতা, ঐতিহ্য) stay plain j-j; not confirmed as a
        # general split yet, so only those two get the override, in the
        # lexicon, not here. ব্য/শ্য (YA_PHALA_INHERENT_A) run the same
        # geminate-non-initial pattern (দৃশ্য "drishsho", উদ্দেশ্য
        # "uddeshsho", দাতব্য "datobbo") -- their vowel then falls through
        # to the normal word-final-conjunct rule below (closed O) instead
        # of YA_PHALA's forced "a", since gemination makes them a real
        # 2-consonant conjunct at that point. ক্য and গ্য run the reverse
        # rule: their override IS the geminated medial/final form (ঐক্য
        # "oikko", যোগ্যতা "joggota"), so word-initial is the exception,
        # staying single (ক্যাপ "kap", not "kkap"; গ্যাস "gas", not "ggas").
        if consonant_str == "জ্ঞ" and not _is_word_initial(clusters, idx):
            phonemes.extend(["g", "g", "y"])
        elif consonant_str == "শ্ব" and not _is_word_initial(clusters, idx):
            phonemes.extend(["sh", "sh"])
        elif consonant_str == "জ্য" and not _is_word_initial(clusters, idx):
            phonemes.extend(["j", "j"])
        elif consonant_str == "হ্য" and not _is_word_initial(clusters, idx):
            phonemes.extend(["j", "j"])
        elif consonant_str == "ব্য" and not _is_word_initial(clusters, idx):
            phonemes.extend(["b", "b"])
        elif consonant_str == "শ্য" and not _is_word_initial(clusters, idx):
            phonemes.extend(["sh", "sh"])
        elif consonant_str == "ক্য" and _is_word_initial(clusters, idx):
            phonemes.append("k")
        elif consonant_str == "গ্য" and _is_word_initial(clusters, idx):
            phonemes.append("g")
        else:
            phonemes.extend(override)
    elif consonant_str == "হ" and cluster.vowel_sign == "ৃ":
        # হ + ঋ-kar: হ is silent here (হৃদয় "ridoy", হৃদরোগ "ridrog") --
        # a single-consonant + vowel-sign pairing, not a conjunct, so it's
        # not in CONJUNCT_OVERRIDES; confirmed against the existing
        # হৃদরোগ/হৃদরোগ বিশেষজ্ঞ lexicon entries, both already হ-less.
        pass
    else:
        consonants_to_render = cluster.consonants
        if (cluster.vowel_sign == "া" and len(cluster.consonants) > 1
                and cluster.consonants[-1] == "য"):
            # trailing ্য here is a silent ya-phala glide marker, not a
            # pronounced consonant -- its only job is to mark the "A"
            # vowel quality below (ফ্ল্যাশ "phlAsh", not "phljAsh")
            consonants_to_render = cluster.consonants[:-1]
        for c in consonants_to_render:
            phonemes.append(CONSONANTS[c])

    if vowel_initial_ya_phala:
        pass  # "A" already appended above
    elif cluster.vowel_sign == "__NONE__":
        pass
    elif (cluster.vowel_sign == "া" and len(cluster.consonants) > 1
            and cluster.consonants[-1] == "য" and _is_word_initial(clusters, idx)):
        # ্যা (ya-phala + aa-kar) is the English "æ" vowel, but only when
        # word-initial (ক্যামেরা "cæmera", শ্যালক "shælok", ফ্ল্যাশ "phlæsh")
        # -- the same spelling elsewhere in a word is plain "a" (সংখ্যা
        # "shongkha"), and ্যো (ya-phala + O-kar) is unaffected even
        # word-initially (জ্যোতি
        # "joti")
        phonemes.append("A")
    elif cluster.vowel_sign == "ে":
        phonemes.append("E" if _eekar_is_open(clusters, idx) else "e")
    elif cluster.vowel_sign is not None:
        phonemes.append(VOWEL_SIGNS[cluster.vowel_sign])
    elif consonant_str == KHANDA_TA:
        # খণ্ড ত never carries an inherent vowel, in any position
        pass
    elif consonant_str in YA_PHALA_INHERENT_A and _is_word_initial(clusters, idx):
        # ব্য/শ্য-family word-initial: inherent vowel is never suppressed by
        # context, and is the "æ" quality (ব্যথা "bætha"). Non-initial is NOT
        # handled here any more -- gemination above makes it a real conjunct,
        # so it falls through to the normal word-final-conjunct rule (closed
        # O) below instead of a forced "a".
        phonemes.append("A")
    elif _schwa_should_keep(clusters, idx):
        if is_word_final(clusters, idx):
            # Word-final retained schwa only happens for a conjunct (Rule 2
            # always drops a lone word-final consonant's schwa) and is
            # always the closed sound: রক্ত "rokto", বিশ্ব "bishsho", not
            # the open "aw". There's no following syllable to check for
            # harmony here, so this is checked ahead of that logic.
            phonemes.append("O")
        elif (len(cluster.consonants) == 1
                and (_next_syllable_triggers_harmony(clusters, idx)
                     or _penultimate_sonorant_triggers(clusters, idx))):
            phonemes.append("O")
        else:
            phonemes.append("o")
    # else: schwa deletion algorithm says drop -- append nothing

    if cluster.nasalized:
        if phonemes:
            phonemes[-1] = phonemes[-1] + "~"

    if cluster.trailing_diacritic:
        phonemes.append(DIACRITIC_PHONEMES[cluster.trailing_diacritic])

    return phonemes


def text_to_phonemes(text: str) -> List[str]:
    text = preprocess(text)   # normalize + fix nukta sequences
    clusters = segment_clusters(text)
    result: List[str] = []
    for idx, cluster in enumerate(clusters):
        if getattr(cluster, "_passthrough", None) is not None:
            ch = cluster._passthrough
            if ch.isspace():
                result.append("|")
            continue
        result.extend(cluster_to_phonemes(cluster, clusters, idx))
    return result


if __name__ == "__main__":
    tests = [
        ("আমরা",      ["a","m","r","a"]),
        ("তোমরা",     ["t","o","m","r","a"]),
        ("সকাল",      ["sh","o","k","a","l"]),
        ("করছি",      ["k","o","r","chh","i"]),
        ("কম",        ["k","o","m"]),
        ("বাংলা",     ["b","a","ng","l","a"]),
        ("জ্ঞান",     ["g","y","a","n"]),
        ("বিজ্ঞান",   ["b","i","g","g","y","a","n"]),
        ("সময়",      ["sh","o","m","o","y"]),
        ("বিষয়",     ["b","i","sh","o","y"]),
        ("হয়",       ["h","o","y"]),
    ]
    all_pass = True
    for word, expected in tests:
        result = text_to_phonemes(word)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{status}  {word}: got {result}, expected {expected}")
    print()
    print("ALL PASS" if all_pass else "SOME FAILED — check above")
