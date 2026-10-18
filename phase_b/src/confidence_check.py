"""
Step 1 of the hybrid-router protocol: does confidence carry signal at all?

Loads the trained checkpoint, runs it on the OFFICIAL val split (not the
fresh val set -- that's reserved for final reporting only), bins words by
greedy-decode confidence, and reports word-exact-match accuracy per bin.

If accuracy does not rise clearly from the lowest to the highest bin,
confidence has no usable signal here and a confidence-based router cannot
help -- stop, don't build the hybrid dataset.

Usage:
    python confidence_check.py --checkpoint ../checkpoints/best_model.pt \
        --val_file ../data/lexicon_split_val.tsv
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from data import load_lexicon, encode_word, PAD, SOS, EOS
from model import Seq2SeqG2P
from normalizer import compose_nukta


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocabs = ckpt["vocabs"]
    args = ckpt["args"]
    pad_id_in = vocabs["input_stoi"][PAD]
    pad_id_out = vocabs["output_stoi"][PAD]
    model = Seq2SeqG2P(
        input_vocab_size=len(vocabs["input_vocab"]),
        output_vocab_size=len(vocabs["output_vocab"]),
        emb_dim=args["emb_dim"],
        hidden_dim=args["hidden_dim"],
        register_vocab_size=len(vocabs["register_stoi"]),
        pad_id_in=pad_id_in,
        pad_id_out=pad_id_out,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="../checkpoints/best_model.pt")
    ap.add_argument("--val_file", default="../data/lexicon_split_val.tsv")
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabs = load_model(args.checkpoint, device)
    input_stoi = vocabs["input_stoi"]
    output_itos = {i: t for t, i in vocabs["output_stoi"].items()}
    register_stoi = vocabs["register_stoi"]
    sos_id = vocabs["output_stoi"][SOS]
    eos_id = vocabs["output_stoi"][EOS]

    entries = load_lexicon(args.val_file)
    print(f"Loaded {len(entries)} words from {args.val_file}")

    rows = []  # (word, confidence, is_correct)
    for word, gold_phonemes, register in entries:
        word = compose_nukta(word)
        input_ids = torch.tensor([encode_word(word, input_stoi)], dtype=torch.long).to(device)
        input_lens = torch.tensor([len(word)], dtype=torch.long)
        register_id = torch.tensor([register_stoi.get(register, 0)], dtype=torch.long).to(device)

        with torch.no_grad():
            preds, confidences = model.greedy_decode(input_ids, input_lens, register_id, sos_id, eos_id)

        pred_tokens = [output_itos.get(t, "?") for t in preds[0]]
        is_correct = pred_tokens == gold_phonemes
        rows.append((word, confidences[0], is_correct))

    overall_acc = sum(r[2] for r in rows) / len(rows)
    print(f"Overall val word-exact-match accuracy: {overall_acc:.3f}\n")

    # Quantile bins (equal population per bin) -- equal-width bins are
    # useless when confidence is this skewed toward 1.0, which it is here.
    rows_sorted = sorted(rows, key=lambda r: r[1])
    n = len(rows_sorted)
    n_bins = args.bins
    print(f"{'bin (conf range)':<22}{'n':>7}{'accuracy':>10}")
    print("-" * 39)
    prev_acc = None
    rising = True
    for b in range(n_bins):
        lo_i = n * b // n_bins
        hi_i = n * (b + 1) // n_bins
        in_bin = rows_sorted[lo_i:hi_i]
        if not in_bin:
            continue
        acc = sum(r[2] for r in in_bin) / len(in_bin)
        lo_c, hi_c = in_bin[0][1], in_bin[-1][1]
        print(f"[{lo_c:.3f}, {hi_c:.3f}]      {len(in_bin):>7}{acc:>10.3f}")
        if prev_acc is not None and acc < prev_acc - 0.02:
            rising = False
        prev_acc = acc

    print()
    if rising:
        print("Accuracy rises (within noise tolerance) from low to high confidence bins.")
        print("-> Confidence carries usable signal. Proceed to threshold selection.")
    else:
        print("Accuracy does NOT rise cleanly across bins.")
        print("-> Confidence does not reliably separate correct from incorrect "
              "predictions here. A confidence-based router would be routing on "
              "noise -- do not build the hybrid on this signal.")


if __name__ == "__main__":
    main()
