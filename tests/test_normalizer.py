"""
Tests for normalizer.py

Run with: python -m pytest tests/ -v

Each test = one thing the normalizer must get right.
When you fix a bug, add a test for it here so it never regresses.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from normalizer import (
    bangla_digits_to_arabic,
    normalize_numbers,
    expand_abbreviations,
    handle_english_words,
    strip_punctuation,
    normalize,
)
from g2p_engine import text_to_phonemes


# ── Digit conversion ──────────────────────────────────────────────────────────

def test_bangla_digits_to_arabic():
    assert bangla_digits_to_arabic("১২৩") == "123"

def test_bangla_digits_mixed():
    assert bangla_digits_to_arabic("আমার ৫ টাকা") == "আমার 5 টাকা"


# ── Number to words ───────────────────────────────────────────────────────────

def test_single_digit_word():
    assert normalize_numbers("5") == "পাঁচ"

def test_twelve_to_bangla():
    assert normalize_numbers("12") == "বারো"

def test_one_hundred():
    # "শো" (explicit vowel sign), not "শত" -- the numeral suffix "-sho" is
    # never pronounced "-shoto", unlike the standalone noun "শত" (century)
    assert normalize_numbers("100") == "একশো"

def test_number_inside_sentence():
    result = normalize_numbers("তার বয়স 25 বছর")
    assert "বছর" in result
    assert "25" not in result

def test_twenty_three_is_irregular():
    # 23 -> "তেইশ", not the literal "বিশ তিন" (twenty-three) -- Bangla
    # numbers 21-99 are each their own irregular word, not compositional
    assert normalize_numbers("23") == "তেইশ"

def test_sixty_seven_is_irregular():
    assert normalize_numbers("67") == "সাতষট্টি"

def test_four_digit_number():
    assert normalize_numbers("3409") == "তিন হাজার চারশো নয়"

def test_exact_hundred_uses_sho_suffix():
    # "চারশো", not "চারশত" -- same "-sho" suffix whether or not a
    # remainder follows
    assert normalize_numbers("400") == "চারশো"

def test_hundred_with_remainder_full_pronunciation():
    # End-to-end (normalize + g2p): "চার" must keep its own standalone
    # schwa-drop ("char", not "charo") and the "-শো" suffix must keep its
    # own vowel -- ch-a-r-sh-O, not ch-a-r-o-sh (the bug this was written
    # against). "O" (not "o") because শো uses the ও vowel sign, which is
    # phonemically distinct from the অ inherent vowel/schwa.
    phonemes = text_to_phonemes(normalize_numbers("409"))
    assert phonemes == ["ch", "a", "r", "sh", "O", "|", "n", "o", "y"]

def test_hundred_exact_full_pronunciation():
    phonemes = text_to_phonemes(normalize_numbers("100"))
    assert phonemes == ["e", "k", "sh", "O"]


# ── Abbreviations ─────────────────────────────────────────────────────────────

def test_doctor_abbreviation():
    assert "ডক্টর" in expand_abbreviations("ডঃ আহমেদ")

def test_dhaka_abbreviation():
    assert "ঢাকা" in expand_abbreviations("ঢাঃ শহর")

def test_mbbs_abbreviation():
    # Latin-script abbreviations must expand to Bangla script here, before
    # handle_english_words() strips remaining Latin letters
    result = normalize("তিনি MBBS পাস করেছেন")
    assert "MBBS" not in result
    assert "এম বি বি এস" in result

def test_hsc_abbreviation():
    result = normalize("সে HSC পরীক্ষা দিয়েছে")
    assert "এইচ এস সি" in result

def test_engr_abbreviation():
    result = normalize("Engr. করিম আসবেন")
    assert "ইঞ্জিনিয়ার" in result


# ── English word removal ──────────────────────────────────────────────────────

def test_english_words_removed():
    result = handle_english_words("আমার mobile আছে")
    assert "mobile" not in result
    assert "আমার" in result

def test_bangla_untouched_by_english_handler():
    text = "আমি ভালো আছি"
    assert handle_english_words(text) == text


# ── Punctuation ───────────────────────────────────────────────────────────────

def test_danda_stripped():
    result = strip_punctuation("বাংলা।")
    assert "।" not in result

def test_comma_stripped():
    result = strip_punctuation("হ্যাঁ, না")
    assert "," not in result


# ── Known bugs (marked xfail until fixed) ────────────────────────────────────
# These are tests for KNOWN bugs. They are expected to fail for now.
# When you fix the bug, remove the xfail marker and the test should pass.

import pytest

@pytest.mark.xfail(reason="Abbreviation 'বাং' incorrectly matches inside বাংলা/বাংলাদেশ — needs word-boundary fix")
def test_abbr_no_false_match_inside_word():
    result = expand_abbreviations("বাংলাদেশ")
    # Should NOT expand বাং inside বাংলাদেশ
    assert result == "বাংলাদেশ"
