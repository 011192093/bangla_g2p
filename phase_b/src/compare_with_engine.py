"""
Phase B — Compare Neural Model vs Phase A Rule Engine
========================================================
Runs both your Phase A rule engine (g2p_engine.py) and the trained
Phase B neural model on the SAME held-out validation words, and
reports which one gets closer to the lexicon ground truth.

This produces the key comparison table for your paper:
    "Phase A rules: X% word-exact-match on held-out words"
    "Phase B neural: Y% word-exact-match on held-out words"

IMPORTANT: point --g2p_src_dir at your actual bangla_g2p/src folder
so this script can import your real text_to_phonemes function.

Usage:
    python compare_with_engine.py \
        --checkpoint ../checkpoints/best_model.pt \
        --lexicon ../data/lexicon.tsv \
        --g2p_src_dir /path/to/bangla_g2p/src
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import load_lexicon, train_val_split
from predict import load_model, predict_word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/best_model.pt")
    parser.add_argument("--lexicon", type=str, default="../data/lexicon_test.tsv")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--g2p_src_dir", type=str, default=None,
                         help="Path to your Phase A src/ folder containing g2p_engine.py")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabs = load_model(args.checkpoint, device)

    entries = load_lexicon(args.lexicon)
    _, val_entries = train_val_split(entries, args.val_ratio, args.seed)
    print(f"Evaluating on {len(val_entries)} held-out words (never seen in training)\n")

    # Try to import the real rule engine if a path was given
    text_to_phonemes = None
    if args.g2p_src_dir:
        sys.path.insert(0, args.g2p_src_dir)
        try:
            from g2p_engine import text_to_phonemes as _ttp
            text_to_phonemes = _ttp
            print(f"Loaded Phase A rule engine from {args.g2p_src_dir}\n")
        except ImportError as e:
            print(f"Could not import g2p_engine from {args.g2p_src_dir}: {e}")
            print("Continuing with neural-model-only evaluation.\n")

    rule_correct = 0
    neural_correct = 0
    both_correct = 0
    neither_correct = 0
    rule_only = 0
    neural_only = 0

    rows = []
    for word, gold_phonemes, register in val_entries:
        neural_pred = predict_word(model, vocabs, word, register, device)
        neural_ok = neural_pred == gold_phonemes

        rule_pred = None
        rule_ok = None
        if text_to_phonemes:
            try:
                rule_pred = text_to_phonemes(word)
                rule_ok = rule_pred == gold_phonemes
            except Exception:
                rule_pred = ["<error>"]
                rule_ok = False

        if neural_ok:
            neural_correct += 1
        if rule_ok:
            rule_correct += 1
        if rule_ok and neural_ok:
            both_correct += 1
        elif rule_ok and not neural_ok:
            rule_only += 1
        elif neural_ok and not rule_ok:
            neural_only += 1
        else:
            neither_correct += 1

        rows.append((word, register, gold_phonemes, rule_pred, rule_ok, neural_pred, neural_ok))

    n = len(val_entries)
    print(f"{'Word':<15}{'Register':<20}{'Gold':<25}{'Rule':<8}{'Neural':<8}")
    print("-" * 80)
    for word, register, gold, rule_pred, rule_ok, neural_pred, neural_ok in rows:
        r_icon = "✅" if rule_ok else ("—" if rule_pred is None else "❌")
        n_icon = "✅" if neural_ok else "❌"
        print(f"{word:<15}{register:<20}{' '.join(gold):<25}{r_icon:<8}{n_icon:<8}")

    print("\n" + "=" * 60)
    print(f"Held-out set size:              {n}")
    if text_to_phonemes:
        print(f"Phase A rule engine accuracy:   {rule_correct}/{n} = {rule_correct/n*100:.1f}%")
    print(f"Phase B neural model accuracy:  {neural_correct}/{n} = {neural_correct/n*100:.1f}%")
    if text_to_phonemes:
        print(f"Both correct:                    {both_correct}")
        print(f"Rule correct, neural wrong:      {rule_only}")
        print(f"Neural correct, rule wrong:      {neural_only}")
        print(f"Both wrong:                      {neither_correct}")
    print("=" * 60)


if __name__ == "__main__":
    main()
