"""
Test suite for the rule-based Bangla G2P engine.

Run with:  pytest tests/ -v

As you find errors from native-speaker review, ADD a test case here for
each one (expected correct output), then fix the engine until it passes.
This file becomes your regression safety net — never let a fixed case
break again silently.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from g2p_engine import text_to_phonemes


def test_simple_word_no_deletion():
    # নাম -> "naam" : multi-letter, vowel sign present, no deletion needed
    assert text_to_phonemes("নাম") == ["n", "a", "m"]


def test_word_final_schwa_deletion():
    # কম -> "kom" : single final consonant drops trailing inherent vowel
    assert text_to_phonemes("কম") == ["k", "o", "m"]


def test_anusvara_nasalization():
    # বাংলা -> "bangla"
    assert text_to_phonemes("বাংলা") == ["b", "a", "ng", "l", "a"]


def test_conjunct_override_kkho():
    # ক্ষমা -> "kkhoma"
    result = text_to_phonemes("ক্ষমা")
    assert result[0] == "kkh"


def test_visarga_to_h():
    # দুঃখ -> contains 'h' from visarga
    result = text_to_phonemes("দুঃখ")
    assert "h" in result


def test_explicit_hasanta_no_vowel():
    # করছি -> "korchhi" in spoken Bangla
    # ক keeps its 'o' (র has no explicit vowel sign)
    # র drops its 'o' (ছি has explicit 'ি' vowel sign)
    result = text_to_phonemes("করছি")
    assert result == ["k", "o", "r", "chh", "i"]


def test_space_becomes_boundary_token():
    result = text_to_phonemes("আমি ভালো")
    assert "|" in result
def test_sa_is_sh():
    # স as standalone consonant should be sh
    result = text_to_phonemes("সে")
    assert result[0] == "sh"

def test_mid_word_schwa_deletion():
    # আমরা — inherent vowel after ম should be dropped before র
    result = text_to_phonemes("আমরা")
    assert result == ["a", "m", "r", "a"]

def test_chandrabindu_after_vowel_sign():
    # পাঁচ -> "pa~ch" : chandrabindu nasalizes a vowel SIGN, not just a
    # standalone vowel or bare consonant. Previously dropped entirely.
    result = text_to_phonemes("পাঁচ")
    assert result == ["p", "a~", "ch"]


def test_ra_with_nukta():
    # পড়া -> "pora" : ড় is a single flap "r", not doubled "rr"
    result = text_to_phonemes("পড়া")
    assert result == ["p", "o", "r", "a"]


def test_rha_with_nukta():
    # আষাঢ় -> "ashar" : ঢ় merges with plain "r", same as ড় -- the
    # aspiration is not audible in practice
    result = text_to_phonemes("আষাঢ়")
    assert result == ["a", "sh", "a", "r"]


def test_murdhanya_na_merges_with_na():
    # লবণ -> "lobOn" : ণ is phonemically merged with ন in modern Bangla.
    # The "O" comes from the penultimate-sonorant rule (see
    # _penultimate_sonorant_triggers), not from this ণ/ন merger itself.
    result = text_to_phonemes("লবণ")
    assert result == ["l", "o", "b", "O", "n"]


def test_ndhya_conjunct_elides_glide():
    # সন্ধ্যা -> "shondha" : the য in this conjunct is not pronounced as a
    # glide here, unlike জ্ঞান-style clusters
    result = text_to_phonemes("সন্ধ্যা")
    assert result == ["sh", "o", "n", "dh", "a"]


def test_shba_conjunct_geminates():
    # বিশ্ব -> "bishshO" : শ্ব geminates ("sh-sh") rather than gliding to "shw"
    result = text_to_phonemes("বিশ্ব")
    assert result == ["b", "i", "sh", "sh", "O"]


def test_nya_conjunct_geminates():
    # বন্যা -> "bonna" : ন্য geminates ("n-n") rather than gliding to "nj"
    result = text_to_phonemes("বন্যা")
    assert result == ["b", "o", "n", "n", "a"]


def test_shma_conjunct_geminates():
    # গ্রীষ্ম -> "grishshO" : ষ্ম geminates ("sh-sh") rather than "sh-m"
    result = text_to_phonemes("গ্রীষ্ম")
    assert result == ["gr", "i", "sh", "sh", "O"]


def test_jba_conjunct_drops_ba():
    # জ্বর -> "jor" : the ব in this conjunct is not pronounced
    result = text_to_phonemes("জ্বর")
    assert result == ["j", "o", "r"]


def test_khanda_ta_word_final():
    # বিদ্যুৎ -> "biddut" : খণ্ড ত (khanda-ta) was previously dropped entirely
    result = text_to_phonemes("বিদ্যুৎ")
    assert result == ["b", "i", "d", "d", "u", "t"]


def test_khanda_ta_never_takes_inherent_vowel():
    # উৎসব -> "utshob" : খণ্ড ত never takes a vowel, even mid-word
    result = text_to_phonemes("উৎসব")
    assert result == ["u", "t", "sh", "o", "b"]


def test_word_initial_resets_after_space():
    # শুভ সকাল -> the স in সকাল is word-initial for schwa purposes even
    # though it isn't the first cluster in the whole string. Previously
    # _is_word_initial scanned back to index 0 instead of the last space,
    # so সকাল wrongly lost its "o" (sh-k-a-l instead of sh-o-k-a-l).
    result = text_to_phonemes("শুভ সকাল")
    assert result == ["sh", "u", "bh", "|", "sh", "o", "k", "a", "l"]


def test_bya_conjunct_inherent_a():
    # ব্যাগ -> "bæg" : ব্য drops the "j" glide entirely; word-initial ্যা
    # is the English "æ" vowel, not plain "a"
    result = text_to_phonemes("ব্যাগ")
    assert result == ["b", "A", "g"]


def test_bya_conjunct_no_explicit_vowel_sign():
    # ব্যথা -> "bætha" : with no explicit vowel sign, ব্য's inherent vowel
    # is never suppressed by neighboring context the way a normal schwa
    # would be, and is "æ" (not "o"/"a") because ব্য is word-initial here
    result = text_to_phonemes("ব্যথা")
    assert result == ["b", "A", "th", "a"]


def test_shya_conjunct():
    # শ্যালক -> "shælok" : শ্য drops the "j" glide, same family as ব্য;
    # word-initial ্যা is "æ"
    result = text_to_phonemes("শ্যালক")
    assert result == ["sh", "A", "l", "o", "k"]


def test_ya_phala_a_not_word_initial():
    # সংখ্যা's খ্য is not word-initial, so it stays plain "a" rather than
    # becoming "æ" -- the vowel-color rule is specifically about
    # word-initial position, not just the presence of ্যা
    result = text_to_phonemes("সংখ্যা")
    assert result == ["sh", "o", "ng", "kh", "a"]


def test_ya_phala_o_kar_unaffected():
    # জ্যোতি -> "joti" : ্যো (ya-phala + O-kar, not aa-kar) is unaffected
    # by the æ rule even word-initially -- only ্যা triggers it. জ্য stays
    # single "j" word-initially, same as জ্ঞ/শ্ব's word-initial behavior.
    result = text_to_phonemes("জ্যোতি")
    assert result == ["j", "O", "t", "i"]


def test_jya_conjunct_geminates_mid_word():
    # বানিজ্য -> "banijjO" : জ্য geminates when not word-initial, same
    # family as জ্ঞ/শ্ব (contrast with জ্যোতি/জ্যৈষ্ঠ/জ্যান্ত above, all
    # word-initial and therefore single "j")
    result = text_to_phonemes("বানিজ্য")
    assert result == ["b", "a", "n", "i", "j", "j", "O"]


def test_hya_conjunct_geminates_mid_word():
    # গ্রাহ্য -> "grajjO" : হ্য geminates to "j-j" (dropping হ entirely,
    # same as হ্ব/হ্ন/হ্ম) when not word-initial -- contrast with হ্যাঁ,
    # which is word-initial and keeps single "h"
    result = text_to_phonemes("গ্রাহ্য")
    assert result == ["gr", "a", "j", "j", "O"]


def test_cha_conjunct_single():
    # চ্যালেঞ্জ -> "chælenjO" : চ্য drops to a single "ch", same single-
    # consonant family as ব্য/শ্য/হ্য
    result = text_to_phonemes("চ্যালেঞ্জ")
    assert result == ["ch", "A", "l", "e", "nj", "O"]


def test_gya_conjunct_geminates():
    # যোগ্যতা -> গ্য geminates to "g-g", same family as ন্য/দ্য/ত্য/ক্য/ল্য
    result = text_to_phonemes("যোগ্যতা")
    assert result == ["j", "O", "g", "g", "t", "a"]


def test_tya_conjunct_geminates():
    # সাহিত্য -> "shahittO" : ত্য geminates ("t-t"), same family as থ্য/দ্য
    result = text_to_phonemes("সাহিত্য")
    assert result == ["sh", "a", "h", "i", "t", "t", "O"]


def test_rghya_conjunct_drops_glide():
    # দৈর্ঘ্য -> "doirghO" : র্ঘ্য drops the য glide entirely
    result = text_to_phonemes("দৈর্ঘ্য")
    assert result == ["d", "oi", "r", "gh", "O"]


def test_shba_word_initial_no_gemination():
    # শ্বশুর -> "shoshur" : শ্ব stays a single "sh" word-initially --
    # gemination needs a preceding vowel to double against. Contrast with
    # medial বিশ্ব, which does geminate ("bishsho").
    result = text_to_phonemes("শ্বশুর")
    assert result == ["sh", "o", "sh", "u", "r"]


def test_hba_conjunct_geminates_ba():
    # জিহ্বা -> "jibba" : হ্ব drops হ entirely and geminates "b"
    result = text_to_phonemes("জিহ্বা")
    assert result == ["j", "i", "b", "b", "a"]


def test_hra_conjunct_drops_ha():
    # হ্রদ -> "rod" : হ্র drops হ entirely, leaving just "r"
    result = text_to_phonemes("হ্রদ")
    assert result == ["r", "o", "d"]


def test_hya_conjunct_drops_glide():
    # হ্যাঁ -> "hæ~" : হ্য drops the য glide, keeping just "h"; word-initial
    # ্যা is "æ"
    result = text_to_phonemes("হ্যাঁ")
    assert result == ["h", "A~"]


def test_kkhma_conjunct_geminates():
    # লক্ষ্মী -> "lOkkhi" : ক্ষ্ম geminates to "kkh", dropping ম entirely.
    # "O" (not "o") because ল's inherent vowel raises via vowel harmony --
    # ী is a high vowel, see test_vowel_harmony_* below
    result = text_to_phonemes("লক্ষ্মী")
    assert result == ["l", "O", "kkh", "i"]


def test_dma_conjunct_geminates():
    # পদ্ম -> "poddO" : দ্ম geminates to "d-d"
    result = text_to_phonemes("পদ্ম")
    assert result == ["p", "o", "d", "d", "O"]


def test_ttba_conjunct_drops_ba():
    # তত্ত্ব -> "tottO" : ত্ত্ব drops the trailing ব entirely
    result = text_to_phonemes("তত্ত্ব")
    assert result == ["t", "o", "t", "t", "O"]


def test_kya_conjunct_geminates():
    # ঐক্য -> "oikkO" : ক্য geminates, no "j" sound
    result = text_to_phonemes("ঐক্য")
    assert result == ["oi", "k", "k", "O"]


def test_kya_conjunct_single_word_initial():
    # ক্যাপ -> "kAp" : ক্য stays single "k" word-initially (loanword
    # ্যা pattern, same as ব্য/শ্য/চ্য/হ্য family) -- contrast with ঐক্য
    # above, which geminates because ক্য isn't word-initial there
    result = text_to_phonemes("ক্যাপ")
    assert result == ["k", "A", "p"]


def test_khya_conjunct_drops_glide():
    # সংখ্যা -> "shongkha" : খ্য drops the য glide entirely
    result = text_to_phonemes("সংখ্যা")
    assert result == ["sh", "o", "ng", "kh", "a"]


def test_kkhya_conjunct_geminates_and_drops_glide():
    # লক্ষ্য -> "lokkhO" : ক্ষ্য geminates to "kkh" and drops the য glide
    result = text_to_phonemes("লক্ষ্য")
    assert result == ["l", "o", "kkh", "O"]


def test_sthya_conjunct_drops_glide():
    # স্বাস্থ্য -> "shashthO" : স্থ্য drops the য glide entirely and
    # dissimilates to a plain "s" (matches স্থ elsewhere, e.g. স্থান)
    result = text_to_phonemes("স্বাস্থ্য")
    assert result == ["sh", "a", "s", "th", "O"]


def test_kkhna_conjunct():
    # তীক্ষ্ণ -> "tikhnO" : ক্ষ্ণ drops the ষ sound, keeping "k-h-n"
    result = text_to_phonemes("তীক্ষ্ণ")
    assert result == ["t", "i", "k", "h", "n", "O"]


def test_shra_conjunct_is_sr():
    # শ্রাবণ -> "srabon" : শ্র is a plain "s-r", not "shr"
    result = text_to_phonemes("শ্রাবণ")
    assert result == ["s", "r", "a", "b", "o", "n"]


def test_shla_conjunct_is_sl():
    # বিশ্লেষণ -> "bisleshon" : শ্ল is a plain "s-l", same family as শ্র
    result = text_to_phonemes("বিশ্লেষণ")
    assert result == ["b", "i", "s", "l", "e", "sh", "o", "n"]


def test_lya_conjunct_geminates():
    # সাফল্যের -> "shapholler" : ল্য geminates, same family as ন্য/দ্য/ত্য/ক্য.
    # ফ before it still drops its schwa: ল্য is a conjunct but carries
    # its own vowel sign (ে), so the schwa-deletion algorithm's "vowel
    # wins over conjunct-ness" rule applies -- a vowel-bearing conjunct
    # behaves like a single consonant with a vowel, not like a bare
    # conjunct (contrast with রক্ত "roktO", where ক্ত has no vowel of its
    # own and the preceding syllable keeps its schwa).
    result = text_to_phonemes("সাফল্যের")
    assert result == ["sh", "a", "ph", "l", "l", "e", "r"]


def test_hna_conjunct_geminates_drops_ha():
    # চিহ্ন -> "chinnO" : হ্ন geminates "n-n", dropping হ, same family as হ্ব
    result = text_to_phonemes("চিহ্ন")
    assert result == ["ch", "i", "n", "n", "O"]


def test_hma_conjunct_geminates_drops_ha():
    # ব্রাহ্মণ -> "brammon" : হ্ম geminates "m-m", dropping হ
    result = text_to_phonemes("ব্রাহ্মণ")
    assert result == ["br", "a", "m", "m", "o", "n"]


# --- Schwa deletion algorithm ---
# Rule-based decision tree for whether a bare consonant's inherent vowel
# is present at all (distinct from vowel *quality*, which harmony
# governs separately). See _schwa_should_keep in g2p_engine.py.

def test_schwa_keep_before_word_final_bare_consonant():
    # শ্যালক -> "shalok" : ল keeps its vowel because the next cluster
    # (ক) is bare AND word-final -- a bare consonant at the very end of
    # a word needs a linking vowel before it, unlike a mid-word one
    result = text_to_phonemes("শ্যালক")
    assert result == ["sh", "A", "l", "o", "k"]


def test_schwa_drop_before_short_vowel():
    # করছি -> "korchhi" : র drops before ছ (has short vowel ি)
    result = text_to_phonemes("করছি")
    assert result == ["k", "o", "r", "chh", "i"]


def test_schwa_keep_before_coda_conjunct():
    # রক্ত -> "roktO" : র keeps before ক্ত, a vowel-less coda conjunct
    # (ends in the stop ত, not a liquid/glide)
    result = text_to_phonemes("রক্ত")
    assert result == ["r", "o", "kt", "O"]


# --- The ে (e-kar) rule ---
# Only the WORD-INITIAL syllable's ে can ever be the open "ae" sound
# (phoneme "E") -- every later syllable's ে is always closed "e", no
# matter what follows. Within that word-initial syllable, ae happens in
# exactly two cases: before an explicit া, or before a bare SONORANT
# carrying its inherent vowel. Everything else is plain e. See
# SONORANT_CONSONANTS in phoneme_inventory.py for the full writeup.

def test_eekar_open_before_explicit_aa():
    # দেখা -> "dEkha" : the next consonant (খ) carries া, and খা is the
    # word's LAST syllable -> E
    result = text_to_phonemes("দেখা")
    assert result == ["d", "E", "kh", "a"]


def test_eekar_open_before_explicit_o_kar_last_syllable():
    # তেরো -> "tEro" : the next consonant (র) carries ো, and রো is the
    # word's LAST syllable -- ো triggers the same way া does, but only
    # because of position, not because ো itself is special
    result = text_to_phonemes("তেরো")
    assert result == ["t", "E", "r", "O"]


def test_eekar_closed_o_kar_not_last_syllable():
    # কেরোসিন -> "kerosin" : the next consonant (র) carries ো, but রো is
    # NOT the last syllable (শিন follows) -- stays closed. Contrast with
    # তেরো above: same ে+consonant+ো shape, different result, because
    # position (not vowel quality) decides.
    result = text_to_phonemes("কেরোসিন")
    assert result == ["k", "e", "r", "O", "sh", "i", "n"]


def test_eekar_closed_aa_not_last_syllable():
    # চেয়ার -> "cheyar" : the next consonant (য়) carries া, but য়া is
    # NOT the last syllable (র follows) -- stays closed, same reasoning
    # as কেরোসিন above. া isn't unconditionally open either.
    result = text_to_phonemes("চেয়ার")
    assert result == ["ch", "e", "y", "a", "r"]


def test_eekar_closed_before_explicit_i():
    # মেশিন -> "meshin" : the next consonant (শ) carries ি, not া, so
    # even though it's an explicit vowel sign, it's plain e (not ae)
    result = text_to_phonemes("মেশিন")
    assert result == ["m", "e", "sh", "i", "n"]


def test_eekar_closed_before_same_vowel():
    # ছেলে -> "chhele" : the next consonant (ল) carries the same vowel
    # ে, not া -> plain e for both syllables (the second is also closed
    # simply for being the second syllable)
    result = text_to_phonemes("ছেলে")
    assert result == ["chh", "e", "l", "e"]


def test_eekar_closed_word_final():
    # সে -> "she" : ে is the very last cluster in the word, nothing
    # follows at all -> plain e
    result = text_to_phonemes("সে")
    assert result == ["sh", "e"]


def test_eekar_closed_before_standalone_vowel():
    # কেউ -> "keu" : a standalone vowel (উ) follows directly -> plain e
    result = text_to_phonemes("কেউ")
    assert result == ["k", "e", "u"]


def test_eekar_open_before_sonorant_inherent_vowel():
    # কেমন -> "kEmon" : ম carries only its inherent vowel, is a
    # sonorant, AND isn't itself word-final (ন follows) -> E
    result = text_to_phonemes("কেমন")
    assert result == ["k", "E", "m", "o", "n"]


def test_eekar_closed_sonorant_but_word_final():
    # তেল -> "tel" : ল carries only its inherent vowel and IS a
    # sonorant, but ল itself ends the word here -- word-finality wins
    # over sonority -> plain e, not ae. Contrast with কেমন above, where
    # the sonorant (ম) has more word after it.
    result = text_to_phonemes("তেল")
    assert result == ["t", "e", "l"]


def test_eekar_closed_before_obstruent_inherent_vowel():
    # দেবতা -> "debota" : ব carries only its inherent vowel and is an
    # obstruent (not a sonorant) -> plain e. Contrast with কেমন above,
    # where the same "bare inherent vowel" position is a sonorant (ম)
    # and gets E instead.
    result = text_to_phonemes("দেবতা")
    assert result == ["d", "e", "b", "t", "a"]


def test_eekar_closed_not_word_initial():
    # হবে -> "hObe" : the ে is on ব, the SECOND cluster of the word, so
    # it's always plain e regardless of what follows (here, nothing
    # does) -- contrast with word-initial ে cases above
    result = text_to_phonemes("হবে")
    assert result == ["h", "O", "b", "e"]


# --- Vowel height harmony ---
# A bare consonant's inherent vowel is normally open ("aw", as in মহা
# "mawha") but raises to the closed "O" sound (phonemically the same
# vowel as ও/ো) when the *next* syllable has an explicit high/front vowel
# sign (ি/ী/ু/ূ/ে). Conjunct clusters are exempt -- only bare consonants
# harmonize. A handful of common words (কলেজ, পরে, ফলে, রংপুর,
# সংবিধান...) contradict this rule outright and are handled as
# lexicon-only exceptions rather than narrowing the rule -- narrowing it
# (dropping ে as a trigger, or exempting anusvara-bearing clusters) broke
# more words than it fixed each time it was tried (করে/হয়ে/মনে/হচ্ছে for
# the ে case, সংগীত for the anusvara case). See conversation history.

def test_vowel_harmony_raises_before_i():
    # মণি -> "mOni" : ি in the next syllable triggers raising
    result = text_to_phonemes("মণি")
    assert result == ["m", "O", "n", "i"]


def test_vowel_harmony_stays_open_before_a():
    # মহা -> "mawha" : া in the next syllable does not trigger raising
    result = text_to_phonemes("মহা")
    assert result == ["m", "o", "h", "a"]


def test_vowel_harmony_stays_open_before_inherent_vowel():
    # রক্ত -> "rawktO" : the next syllable (ক্ত) is itself inherent/schwa,
    # which does not trigger raising -- র stays open ("o"). The trailing
    # "O" comes from the separate word-final-conjunct rule, not harmony.
    result = text_to_phonemes("রক্ত")
    assert result == ["r", "o", "kt", "O"]


def test_vowel_harmony_raises_when_inherent_neighbor_has_high_vowel():
    # রক্তিম -> "roktim" : contrast with রক্ত above -- here ক্ত is
    # followed by ি, so র's vowel raises
    result = text_to_phonemes("রক্তিম")
    assert result == ["r", "O", "kt", "i", "m"]


def test_vowel_harmony_raises_before_u():
    # হলুদ -> "holud" : ু triggers raising
    result = text_to_phonemes("হলুদ")
    assert result == ["h", "O", "l", "u", "d"]


def test_vowel_harmony_raises_before_e():
    # হবে -> "hObe" : ে triggers raising, same as ি/ী/ু/ূ
    result = text_to_phonemes("হবে")
    assert result == ["h", "O", "b", "e"]


def test_vowel_harmony_exempts_conjuncts():
    # গ্রহ -> "groho" : গ্র is a 2-consonant conjunct, not a bare
    # consonant, so it does not harmonize even though হ's vowel isn't high
    result = text_to_phonemes("গ্রহ")
    assert result == ["gr", "o", "h"]


# --- Penultimate sonorant raising ---
# A second, independent trigger for the closed "O" sound, distinct from
# high-vowel harmony above: in a 3+ syllable word, a bare consonant
# raises when it's immediately followed by a word-final SONORANT
# (ম/ন/ণ/ল/র/ড়) carrying only its own inherent vowel, AND the syllable
# before it is a bare single consonant with an inherent or া vowel.

def test_penultimate_sonorant_raises():
    # গরম -> "gorOm" : র is followed by word-final sonorant ম, preceded
    # by bare-inherent-vowel গ
    result = text_to_phonemes("গরম")
    assert result == ["g", "o", "r", "O", "m"]


def test_penultimate_sonorant_raises_after_aa_kar():
    # ছাগল -> "chhagOl" : গ is followed by word-final sonorant ল,
    # preceded by ছ with া (still counts as "open")
    result = text_to_phonemes("ছাগল")
    assert result == ["chh", "a", "g", "O", "l"]


def test_penultimate_sonorant_needs_three_syllables():
    # তেল -> "tel" : only 2 syllables (ত, ল), so the penultimate-sonorant
    # rule can't apply here at all -- this is closed for the separate
    # eekar reason (see test_eekar_closed_sonorant_but_word_final)
    result = text_to_phonemes("তেল")
    assert result == ["t", "e", "l"]


def test_penultimate_sonorant_exempts_conjunct_preceding_syllable():
    # শ্রাবণ -> "srabon" : ব is followed by word-final sonorant ণ, but
    # the preceding syllable (শ্র) is a 2-consonant conjunct, not bare
    # -- rule doesn't apply, stays unraised
    result = text_to_phonemes("শ্রাবণ")
    assert result == ["s", "r", "a", "b", "o", "n"]


def test_penultimate_sonorant_exempts_eekar_preceding_syllable():
    # কেমন -> "kEmon" : ম is followed by word-final sonorant ন, but the
    # preceding syllable's vowel is ে, not inherent/া -- rule doesn't
    # apply here (ম stays "o"); the ে itself is open for an unrelated
    # reason (see the eekar rule)
    result = text_to_phonemes("কেমন")
    assert result == ["k", "E", "m", "o", "n"]


# --- Word-final retained schwa is always closed (O) ---
# Rule 2 only ever keeps a word-final schwa for a conjunct (a lone
# word-final consonant always drops it). That retained vowel is always
# the closed sound, not the open "aw" -- there's no following syllable
# to check for harmony, so this is checked ahead of harmony in
# cluster_to_phonemes. Single-consonant word-final exceptions that keep
# a vowel against Rule 2's default (মাংস, কত, বড়...) are hand-authored
# lexicon overrides, not engine output, and are fixed directly in
# lexicon_seed.tsv.

def test_word_final_conjunct_schwa_is_closed():
    # তথ্য -> থ্য is a word-final conjunct, retained schwa is O. This only
    # tests that claim, not the whole word's pronunciation -- the verified
    # lexicon has তথ্য as "t O t th O" (first vowel closed too), which the
    # engine doesn't produce and has no known rule for; that's a separate,
    # still-open gap (see the o/O case-swap category in the accuracy audit).
    result = text_to_phonemes("তথ্য")
    assert result == ["t", "o", "t", "th", "O"]


def test_word_final_conjunct_schwa_closed_another_word():
    # শব্দ -> "shobdO" : another word-final conjunct, same closed vowel
    result = text_to_phonemes("শব্দ")
    assert result == ["sh", "o", "b", "d", "O"]


# --- ট্য conjunct geminates, like the other -্য family members ---
# ন্য/জ্য/ত্য/দ্য/ল্য all geminate (double the consonant) rather than
# pronouncing the য as "j" -- ট্য was missing from CONJUNCT_OVERRIDES and
# fell through to the default per-letter expansion ("T", "j") instead.
# Found via চিত্রনাট্য (already lexicon-overridden to "...T-T-O") and
# নৃত্যনাট্য both wanting the same "T T" ending -- two independent words
# hitting the same gap confirmed it's a missing rule, not a one-off.

def test_tya_conjunct_geminates():
    # নৃত্যনাট্য -> "nrittonaTTO" : word-final ট্য geminates like ন্য/জ্য/ত্য
    result = text_to_phonemes("নৃত্যনাট্য")
    assert result == ["n", "ri", "t", "t", "O", "n", "a", "T", "T", "O"]


# --- ট্য conjunct geminates, like the other -্য family members ---
# ন্য/জ্য/ত্য/দ্য/ল্য all geminate (double the consonant) rather than
# pronouncing the য as "j" -- ট্য was missing from CONJUNCT_OVERRIDES and
# fell through to the default per-letter expansion ("T", "j") instead.
# Found via চিত্রনাট্য and নৃত্যনাট্য, both wanting a "T T" ending;
# নাট্য alone isolates just the conjunct fix, since compound words like
# চিত্রনাট্য/নৃত্যনাট্য also hit the separate compound-word schwa-suppression
# issue (see সময়সূচি/কবরস্থান-style cases) and stay lexicon overrides for that.

def test_tya_conjunct_geminates():
    # নাট্য -> "naTTO" : word-final ট্য geminates like ন্য/জ্য/ত্য
    result = text_to_phonemes("নাট্য")
    assert result == ["n", "a", "T", "T", "O"]


# --- হ is silent before ঋ-kar ---
# হৃদয় "ridoy", হৃদরোগ "ridrog" -- this is a single consonant + vowel-sign
# pairing, not a conjunct, so it doesn't go through CONJUNCT_OVERRIDES.
# Generalized from two lexicon entries (হৃদরোগ, হৃদরোগ বিশেষজ্ঞ) that had
# independently been hand-overridden to drop the হ.

def test_ho_silent_before_ri_kar():
    result = text_to_phonemes("হৃদয়")
    assert result == ["ri", "d", "o", "y"]


# --- গ্য stays single word-initial, geminates elsewhere ---
# Same ক্য-style exception (override IS the geminated form) -- গ্য was
# missing the word-initial carve-out, so গ্যাস "gas" came out "ggas".
# যোগ্যতা (already in the lexicon) confirms mid-word gemination is correct.

def test_gya_conjunct_single_word_initial():
    result = text_to_phonemes("গ্যাস")
    assert result == ["g", "A", "sh"]


def test_gya_conjunct_geminates_mid_word():
    result = text_to_phonemes("যোগ্যতা")
    assert result == ["j", "O", "g", "g", "t", "a"]


# --- ব্য/শ্য (YA_PHALA_INHERENT_A) geminate like their siblings, non-initial ---
# Previously always single ("b","a"/"sh","a") regardless of position, per the
# family's "inherent vowel never suppressed" rule -- but দৃশ্য/উদ্দেশ্য were
# already hand-verified in the lexicon as geminated ("...sh-sh-O", not
# "...sh-a"), and কর্তব্য/দাতব্য confirmed the same for ব্য. Non-initial now
# geminates and falls through to the normal word-final-conjunct vowel (O),
# instead of the forced word-initial "a"/"A". Word-initial is unaffected
# (ব্যথা "bætha" stays single, তেi is the one YA_PHALA still forces).

def test_shya_geminates_non_initial():
    result = text_to_phonemes("দৃশ্য")
    assert result == ["d", "ri", "sh", "sh", "O"]


def test_bya_geminates_non_initial():
    result = text_to_phonemes("কর্তব্য")
    assert result == ["k", "o", "r", "t", "o", "b", "b", "O"]


def test_bya_stays_single_word_initial():
    # ব্যথা -> "bætha" : word-initial ব্য is unaffected by the gemination fix
    result = text_to_phonemes("ব্যথা")
    assert result == ["b", "A", "th", "a"]


# --- KNOWN ISSUES / TODO: add a failing test here when you find a bug ---
# Example template:
# def test_jnan_conjunct():
#     # জ্ঞান should be "gyan" -- VERIFY with native speaker, current output
#     # is an approximation and likely needs correction in CONJUNCT_OVERRIDES
#     result = text_to_phonemes("জ্ঞান")
#     assert result == [...]  # fill in once verified
