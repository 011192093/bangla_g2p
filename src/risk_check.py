"""
Risk-flagging for newly added lexicon words.

Scans each word against the gotcha patterns discovered this project
(missing schwa before a heavy cluster, unrecognized conjuncts defaulting
to raw per-character combination, word-final single-consonant schwa
retention candidates, and the স loanword s/sh ambiguity) so review can
focus on the words most likely to be wrong, instead of re-checking
everything by hand.

Run with: python src\\risk_check.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from phoneme_inventory import CONJUNCT_OVERRIDES, HASANTA, classify_char
from g2p_engine import (
    preprocess, segment_clusters, is_word_final, _is_word_initial,
    _next_cluster_is_consonant, cluster_to_orthographic_string,
)


def find_risks(word: str):
    """Return a list of (risk_code, detail) tuples for a single word."""
    risks = []
    text = preprocess(word)
    clusters = segment_clusters(text)

    for idx, c in enumerate(clusters):
        if getattr(c, "_passthrough", None) is not None:
            continue

        if c.standalone_vowel is not None:
            continue

        consonant_str = cluster_to_orthographic_string(c.consonants)

        # Unrecognized conjunct: 2+ consonants joined by hasanta, not in
        # our override table -- falls back to naive per-character
        # combination, which has been wrong more often than right this
        # session (ন্য, ব্য, হ্ন, ইত্যাদি all needed overrides).
        if len(c.consonants) > 1 and consonant_str not in CONJUNCT_OVERRIDES:
            risks.append(("unrecognized_conjunct", consonant_str))

        # Word-final bare single consonant: drops its schwa by default,
        # but tatsama words often retain it (মাংস, শত, কত, বড়...).
        if (c.vowel_sign is None and is_word_final(clusters, idx)
                and len(c.consonants) == 1):
            risks.append(("word_final_schwa_drop", consonant_str))

        # Suppressed schwa before the next syllable's own vowel: our
        # single most common bug this session (পুরনো, চট্টগ্রাম, করলা,
        # ওপরে...). Some suppressions are correct (আমরা, করছি already
        # verified), so this over-flags rather than risks missing one.
        if (c.vowel_sign is None and len(c.consonants) == 1
                and not is_word_final(clusters, idx)
                and _next_cluster_is_consonant(clusters, idx)):
            risks.append(("suppressed_schwa", consonant_str))

        # স anywhere: could be native "sh" or loanword "s" -- genuinely
        # unpredictable from spelling (সৃষ্টি/স্পর্শ are native but still "s").
        if "স" in c.consonants:
            risks.append(("sa_sh_or_s_ambiguous", consonant_str))

    return risks


RISK_LABELS = {
    "unrecognized_conjunct": "Unrecognized conjunct — using raw default, unverified",
    "word_final_schwa_drop": "Word-final consonant — may need trailing vowel retained",
    "suppressed_schwa": "Schwa suppressed before next syllable — may need it back",
    "sa_sh_or_s_ambiguous": "Contains স — could be loanword \"s\" or native \"sh\"",
}


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    test_words = ["পুরনো", "মাংস", "সাইকেল", "আমি"]
    for w in test_words:
        print(w, find_risks(w))
