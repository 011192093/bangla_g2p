"""
Phase B — Inference
=====================
Load a trained checkpoint and predict phonemes for new Bangla words,
including words NOT in the training lexicon — this is the real test
of whether the model learned generalizable patterns (loanword s/sh,
o/O harmony) rather than just memorizing the lexicon.

Usage:
    python predict.py --checkpoint ../checkpoints/best_model.pt --word "নতুন_শব্দ"
    python predict.py --checkpoint ../checkpoints/best_model.pt --interactive
"""

import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import encode_word, PAD, SOS, EOS
from model import Seq2SeqG2P


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


def predict_word(model, vocabs, word, register="unknown", device="cpu"):
    input_stoi = vocabs["input_stoi"]
    output_itos = {i: t for t, i in vocabs["output_stoi"].items()}
    register_stoi = vocabs["register_stoi"]

    input_ids = torch.tensor([encode_word(word, input_stoi)], dtype=torch.long).to(device)
    input_lens = torch.tensor([len(word)], dtype=torch.long)
    register_id = torch.tensor([register_stoi.get(register, 0)], dtype=torch.long).to(device)

    sos_id = vocabs["output_stoi"][SOS]
    eos_id = vocabs["output_stoi"][EOS]

    with torch.no_grad():
        preds = model.greedy_decode(input_ids, input_lens, register_id, sos_id, eos_id)

    tokens = [output_itos.get(t, "?") for t in preds[0]]
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/best_model.pt")
    parser.add_argument("--word", type=str, default=None)
    parser.add_argument("--register", type=str, default="unknown")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabs = load_model(args.checkpoint, device)
    print(f"Loaded model from {args.checkpoint}")
    print(f"Known registers: {list(vocabs['register_stoi'].keys())}\n")

    if args.word:
        tokens = predict_word(model, vocabs, args.word, args.register, device)
        print(f"{args.word} -> {' '.join(tokens)}")

    if args.interactive or not args.word:
        print("Interactive mode. Type a Bangla word, optionally 'word|register'.")
        print("Type 'quit' to exit.\n")
        while True:
            raw = input("Word (or word|register): ").strip()
            if raw.lower() in ("quit", "exit", "q"):
                break
            if not raw:
                continue
            if "|" in raw:
                word, register = raw.split("|", 1)
                word, register = word.strip(), register.strip()
            else:
                word, register = raw, "unknown"
            tokens = predict_word(model, vocabs, word, register, device)
            print(f"  -> {' '.join(tokens)}\n")


if __name__ == "__main__":
    main()
