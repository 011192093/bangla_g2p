"""
Real-World Out-of-Distribution Test
=====================================
Your held-out val_word_acc (79.5%) measures words drawn from the SAME
source pool as your training data (Google's lexicon). It does NOT tell
you how the model performs on genuinely novel words from a different
distribution entirely -- new text, domain-specific terms, or just words
that happen to not be in that specific 65,000-word source.

This script:
  1. Takes real running text (e.g. news, Wikipedia, your own domain)
  2. Finds words NOT in your lexicon at all (genuinely unseen)
  3. Runs them through your ACTUAL trained model to get real predictions
  4. Outputs them for you to verify by ear -- giving a REAL measured
     accuracy on truly out-of-distribution words, not an estimate

Usage (run from phase_b/src, with your venv active):
    python test_ood_words.py --lexicon ../data/lexicon.tsv \
        --checkpoint ../checkpoints/best_model.pt \
        --text ../../bangla_sample_text.txt \
        --output ../data/ood_test_results.tsv

By default, OOD words receive a register from the existing heuristic
classifier. Use --register_mode unknown to reproduce the old baseline.
"""

import sys
import os
import re
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))


def load_lexicon_words(path):
    words = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 1 and parts[0].strip():
                words.add(parts[0].strip())
    return words


def tokenize_bangla_text(text):
    return re.findall(r'[\u0980-\u09FF]+', text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True,
                         help="Path to a .txt file of real Bangla running text")
    parser.add_argument("--domain", default="unspecified",
                         help="Domain label written into the result file, e.g. news or literary")
    parser.add_argument("--output", default="ood_test_results.tsv")
    parser.add_argument("--max_words", type=int, default=100,
                         help="Cap how many unique OOD words to test (by frequency, most common first)")
    parser.add_argument("--register_mode", choices=("classifier", "unknown"),
                         default="classifier",
                         help="Register input for the model: use the heuristic classifier by default, or unknown for the baseline")
    args = parser.parse_args()

    lexicon_words = load_lexicon_words(args.lexicon)
    print(f"Lexicon size: {len(lexicon_words)} words")

    with open(args.text, encoding='utf-8') as f:
        text = f.read()
    tokens = tokenize_bangla_text(text)

    ood_counter = Counter(t for t in tokens if t not in lexicon_words)
    ood_words = [w for w, c in ood_counter.most_common(args.max_words)]

    print(f"Text has {len(tokens)} tokens, {len(set(tokens))} unique")
    print(f"Out-of-distribution (not in lexicon) unique words: {len(ood_counter)}")
    print(f"Testing top {len(ood_words)} most frequent OOD words with the trained model\n")

    # Load your actual trained model -- reuses your existing predict.py
    from predict import load_model, predict_word
    from convert_lexicon import classify_register
    from g2p_engine import text_to_phonemes
    from normalizer import normalize
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabs = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}\n")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# domain\tword\tfrequency_in_text\tregister\tmodel_prediction\tYOUR_VERIFICATION(correct/wrong)\n")
        for word in ood_words:
            if args.register_mode == "classifier":
                rule_tokens = text_to_phonemes(normalize(word))
                register = classify_register(word, " ".join(rule_tokens))
            else:
                register = "unknown"
            tokens_pred = predict_word(model, vocabs, word, register=register, device=device)
            pred_str = " ".join(tokens_pred)
            freq = ood_counter[word]
            f.write(f"{args.domain}\t{word}\t{freq}\t{register}\t{pred_str}\t\n")

    print(f"Results written to {args.output}")
    print("\nNow: open that file, say each word aloud, and mark whether the")
    print("model's prediction is correct or wrong in the last column.")
    print("Count correct/total when done -- THAT is your real, measured")
    print("out-of-distribution accuracy, not an estimate.")


if __name__ == "__main__":
    main()
