"""
Full Bangla TTS Text Processing Pipeline
=========================================
Combines the normalizer and G2P engine into one call.

Usage:
    python src\pipeline.py

Or import in your own code:
    from pipeline import bangla_text_to_phonemes
    phonemes = bangla_text_to_phonemes("ডঃ আহমেদ ১২ জানুয়ারি জন্মগ্রহণ করেন।")
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from normalizer import normalize
from g2p_engine import text_to_phonemes


def bangla_text_to_phonemes(raw_text: str, verbose: bool = False) -> list:
    """
    Full pipeline: raw Bangla text -> list of phoneme tokens

    Args:
        raw_text:  any Bangla text, may contain digits/abbreviations/English
        verbose:   if True, prints each step so you can debug what goes wrong

    Returns:
        List of phoneme strings, e.g. ['b', 'a', 'ng', 'l', 'a']
    """
    if verbose:
        print(f"[1] RAW      : {raw_text}")

    normalized = normalize(raw_text)
    if verbose:
        print(f"[2] NORMALIZED: {normalized}")

    phonemes = text_to_phonemes(normalized)
    if verbose:
        print(f"[3] PHONEMES  : {phonemes}")

    return phonemes


if __name__ == "__main__":
    tests = [
        "বাংলা আমার মাতৃভাষা।",
        "আমি ঢাকায় থাকি।",
        "তার বয়স ২৫ বছর।",
        "ডঃ করিম একজন বিজ্ঞানী।",
    ]

    for sentence in tests:
        print("─" * 50)
        bangla_text_to_phonemes(sentence, verbose=True)
    print("─" * 50)
