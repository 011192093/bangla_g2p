"""Compare greedy decoding with beam search on selected words."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import encode_word, EOS, PAD, SOS
from predict import load_model


def predict_both(model, vocabs, word, register, device, beam_width=5):
    input_stoi = vocabs["input_stoi"]
    output_itos = {i: token for token, i in vocabs["output_stoi"].items()}
    register_stoi = vocabs["register_stoi"]

    input_ids = torch.tensor(
        [encode_word(word, input_stoi)], dtype=torch.long, device=device
    )
    input_lens = torch.tensor([len(word)], dtype=torch.long)
    register_ids = torch.tensor(
        [register_stoi.get(register, 0)], dtype=torch.long, device=device
    )
    sos_id = vocabs["output_stoi"][SOS]
    eos_id = vocabs["output_stoi"][EOS]

    with torch.no_grad():
        greedy_result = model.greedy_decode(
            input_ids, input_lens, register_ids, sos_id, eos_id
        )
        beam_result = model.beam_search_decode(
            input_ids, input_lens, register_ids, sos_id, eos_id,
            beam_width=beam_width,
        )

    greedy_tokens = [output_itos.get(token_id, "?") for token_id in greedy_result[0]]
    beam_tokens = [output_itos.get(token_id, "?") for token_id in beam_result[0]]
    return greedy_tokens, beam_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints/best_model.pt")
    parser.add_argument("--words", nargs="+", required=True)
    parser.add_argument("--register", default="unknown")
    parser.add_argument("--beam_width", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabs = load_model(args.checkpoint, device)

    print(f"{'Word':<18}{'Greedy':<30}"
          f"{'Beam (width=' + str(args.beam_width) + ')':<30}Same?")
    print("-" * 90)

    for word in args.words:
        greedy, beam = predict_both(
            model, vocabs, word, args.register, device, args.beam_width
        )
        greedy_str = " ".join(greedy)
        beam_str = " ".join(beam)
        same = "same" if greedy_str == beam_str else "DIFFERENT"
        print(f"{word:<18}{greedy_str:<30}{beam_str:<30}{same}")


if __name__ == "__main__":
    main()