"""
Interactive runner for the Bangla G2P pipeline.

Run this file and type any Bangla sentence.
Type 'quit' to exit.

Usage:
    python src\run.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from normalizer import normalize
from g2p_engine import text_to_phonemes


def show_output(raw_text: str):
    """Run the full pipeline and print every step clearly."""

    print()
    print("=" * 55)
    print(f"  INPUT      : {raw_text}")

    # Step 1: normalize
    cleaned = normalize(raw_text)
    print(f"  NORMALIZED : {cleaned}")

    # Step 2: G2P
    phonemes = text_to_phonemes(cleaned)
    print(f"  PHONEMES   : {phonemes}")

    # Step 3: readable pronunciation (join with dashes)
    readable = " - ".join(phonemes)
    print(f"  READ AS    : {readable}")
    print("=" * 55)


def main():
    print()
    print("╔══════════════════════════════════════════╗")
    print("║   Bangla G2P - Interactive Mode          ║")
    print("║   Type a Bangla sentence, press Enter.   ║")
    print("║   Type  quit  to exit.                   ║")
    print("╚══════════════════════════════════════════╝")

    while True:
        print()
        user_input = input("  ➤ Type Bangla here: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye!\n")
            break

        if not user_input:
            print("  (empty input, try again)")
            continue

        show_output(user_input)


if __name__ == "__main__":
    main()
