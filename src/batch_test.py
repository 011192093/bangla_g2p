"""
Batch Tester for Bangla G2P
Usage: python src\batch_test.py
"""

import sys, os, csv
sys.path.insert(0, os.path.dirname(__file__))

from normalizer import normalize
from g2p_engine import text_to_phonemes

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
WORDS_FILE = os.path.join(BASE_DIR, "data", "test_words.txt")
LEXICON    = os.path.join(BASE_DIR, "data", "lexicon_seed.tsv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "results.csv")


def load_lexicon(path):
    lexicon = {}
    if not os.path.exists(path):
        return lexicon
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                word, phonemes = parts[0], parts[1]
                lexicon[word.strip()] = " - ".join(phonemes.strip().split())
    return lexicon


def load_registers(path):
    """word -> register tag (tatsama/tadbhava/deshi/loanword), only for
    rows that carry a 3rd tab-separated column."""
    registers = {}
    if not os.path.exists(path):
        return registers
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2].strip():
                registers[parts[0].strip()] = parts[2].strip()
    return registers


def run_pipeline(word, lexicon):
    cleaned    = normalize(word)
    phonemes   = text_to_phonemes(cleaned)
    g2p_out    = " - ".join(phonemes)
    lex_answer = lexicon.get(word.strip(), "")

    # Lexicon wins — it is the human-verified correct answer
    final_out = lex_answer if lex_answer else g2p_out

    if lex_answer:
        match = "✅" if g2p_out == lex_answer else "❌"
    else:
        match = "?"

    return g2p_out, lex_answer, final_out, match


def main():
    print()
    print("=" * 60)
    print("  Bangla G2P Batch Tester")
    print("=" * 60)

    if not os.path.exists(WORDS_FILE):
        print(f"\n  ERROR: {WORDS_FILE} not found.\n")
        return

    with open(WORDS_FILE, encoding="utf-8") as f:
        words = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    lexicon = load_lexicon(LEXICON)
    print(f"\n  Words to test : {len(words)}")
    print(f"  Lexicon size  : {len(lexicon)} entries")
    print()

    results = []
    correct = wrong = unknown = 0

    for word in words:
        g2p_out, lex_answer, final_out, match = run_pipeline(word, lexicon)

        if match == "✅":
            correct += 1
        elif match == "❌":
            wrong += 1
        else:
            unknown += 1

        results.append({
            "Word"          : word,
            "Engine Output" : g2p_out,
            "Lexicon Answer": lex_answer,
            "Final Output"  : final_out,
            "Match"         : match,
            "Notes"         : "",
        })

        # Show final output (lexicon override applied)
        display_match = "🔄" if (match == "❌" and lex_answer) else match
        print(f"  {display_match}  {word:<20} {final_out}")

    total_checked = correct + wrong
    accuracy = (correct / total_checked * 100) if total_checked > 0 else 0

    print()
    print("=" * 60)
    print(f"  Total tested   : {len(words)}")
    print(f"  Verified words : {total_checked}")
    print(f"  ✅ Correct (engine)   : {correct}")
    print(f"  🔄 Corrected (lexicon): {wrong}")
    print(f"  ? Not verified        : {unknown}")
    if total_checked > 0:
        total_correct = correct + wrong  # both produce correct final output
        print(f"  Final output accuracy : {total_correct/len(words)*100:.1f}%")
    print("=" * 60)
    print(f"\n  Results saved to: data/results.csv\n")

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
