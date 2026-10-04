"""
Phase B — Training Script
==========================
Trains the seq2seq G2P model on your lexicon, tracking:
  - Token-level accuracy (per-phoneme correctness)
  - Word-level exact-match accuracy (the metric that matters most —
    matches your batch_test.py "Match" column methodology)

Loads a FIXED, pre-generated train/val/test split (see make_fixed_split.py)
instead of re-shuffling the lexicon on every run -- val/test accuracy is
only meaningful if the same words are held out every time. Vocabulary is
built from TRAIN ONLY, so val/test never leak into what the model's input
and output token sets look like.

Usage:
    python make_fixed_split.py --lexicon ../data/lexicon.tsv   # once
    python train.py --epochs 60
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial

sys.path.insert(0, os.path.dirname(__file__))
from data import (
    load_lexicon, build_vocabs, G2PDataset, collate_batch,
    PAD, SOS, EOS,
)
from model import Seq2SeqG2P


def decode_output(ids, itos, eos_id, pad_id):
    tokens = []
    for i in ids:
        if i == eos_id or i == pad_id:
            break
        tokens.append(itos[i])
    return tokens


def evaluate(model, loader, vocabs, device):
    """
    Returns (token_accuracy, word_exact_match_accuracy, examples)
    word_exact_match is the metric to headline — it mirrors your
    batch_test.py definition of "correct".
    """
    model.eval()
    output_itos = {i: t for t, i in vocabs["output_stoi"].items()}
    sos_id = vocabs["output_stoi"][SOS]
    eos_id = vocabs["output_stoi"][EOS]
    pad_id = vocabs["output_stoi"][PAD]

    total_tokens, correct_tokens = 0, 0
    total_words, correct_words = 0, 0
    examples = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            input_lens = batch["input_lens"]
            output_ids = batch["output_ids"].to(device)
            register_ids = batch["register_ids"].to(device)
            words = batch["words"]

            preds = model.greedy_decode(
                input_ids, input_lens, register_ids, sos_id, eos_id
            )

            for i in range(len(words)):
                gold_ids = output_ids[i].tolist()
                gold_tokens = decode_output(gold_ids[1:], output_itos, eos_id, pad_id)
                pred_tokens = [output_itos.get(t, "?") for t in preds[i]]

                total_words += 1
                is_match = pred_tokens == gold_tokens
                if is_match:
                    correct_words += 1

                for j in range(max(len(gold_tokens), len(pred_tokens))):
                    total_tokens += 1
                    g = gold_tokens[j] if j < len(gold_tokens) else None
                    p = pred_tokens[j] if j < len(pred_tokens) else None
                    if g == p:
                        correct_tokens += 1

                if len(examples) < 10:
                    examples.append((words[i], gold_tokens, pred_tokens, is_match))

    token_acc = correct_tokens / max(1, total_tokens)
    word_acc = correct_words / max(1, total_words)
    return token_acc, word_acc, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="../data/lexicon_split_train.tsv")
    parser.add_argument("--val_file", type=str, default="../data/lexicon_split_val.tsv")
    parser.add_argument("--test_file", type=str, default="../data/lexicon_split_test.tsv")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints")
    parser.add_argument("--teacher_forcing", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)

    # ── Load the fixed split (generated once by make_fixed_split.py) ────
    for p, name in [(args.train_file, "train"), (args.val_file, "val"), (args.test_file, "test")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{name} split file not found: {p}\n"
                f"Run make_fixed_split.py once to generate lexicon_split_{{train,val,test}}.tsv"
            )

    train_entries = load_lexicon(args.train_file)
    val_entries = load_lexicon(args.val_file)
    test_entries = load_lexicon(args.test_file)
    print(f"Train: {len(train_entries)}   Val (held-out): {len(val_entries)}   "
          f"Test (held-out, evaluated once at the end): {len(test_entries)}")
    if len(train_entries) < 20:
        print("WARNING: very small lexicon. Results will not be meaningful "
              "until you plug in your real ~2000-word file.")

    # Vocabulary is built from TRAIN ONLY -- val/test must never influence
    # what tokens the model knows about, or their accuracy isn't a clean
    # measure of generalization.
    vocabs = build_vocabs(train_entries)
    print(f"Input vocab size (Bangla chars): {len(vocabs['input_vocab'])}")
    print(f"Output vocab size (phonemes):    {len(vocabs['output_vocab'])}")

    train_ds = G2PDataset(train_entries, vocabs)
    val_ds = G2PDataset(val_entries, vocabs)
    test_ds = G2PDataset(test_entries, vocabs)

    pad_id_in = vocabs["input_stoi"][PAD]
    pad_id_out = vocabs["output_stoi"][PAD]
    collate = partial(collate_batch, pad_id_in=pad_id_in, pad_id_out=pad_id_out)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate)

    # ── Model ──────────────────────────────────────────────────────────
    model = Seq2SeqG2P(
        input_vocab_size=len(vocabs["input_vocab"]),
        output_vocab_size=len(vocabs["output_vocab"]),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        register_vocab_size=len(vocabs["register_stoi"]),
        pad_id_in=pad_id_in,
        pad_id_out=pad_id_out,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id_out)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_word_acc = 0.0

    # ── Training loop ──────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            input_lens = batch["input_lens"]
            output_ids = batch["output_ids"].to(device)
            register_ids = batch["register_ids"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, input_lens, output_ids, register_ids,
                            teacher_forcing_ratio=args.teacher_forcing)

            # logits: (batch, out_len, vocab)  output_ids: (batch, out_len)
            loss = criterion(
                logits[:, 1:, :].reshape(-1, logits.size(-1)),
                output_ids[:, 1:].reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        if epoch % 5 == 0 or epoch == args.epochs:
            token_acc, word_acc, examples = evaluate(model, val_loader, vocabs, device)
            print(f"Epoch {epoch:3d}  loss={avg_loss:.4f}  "
                  f"val_token_acc={token_acc:.3f}  val_word_acc={word_acc:.3f}")

            if word_acc > best_word_acc:
                best_word_acc = word_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "vocabs": vocabs,
                    "args": vars(args),
                }, os.path.join(args.checkpoint_dir, "best_model.pt"))
        else:
            print(f"Epoch {epoch:3d}  loss={avg_loss:.4f}")

    print(f"\nBest held-out VAL word-exact-match accuracy: {best_word_acc:.3f}")
    print(f"Checkpoint saved to {args.checkpoint_dir}/best_model.pt")

    # Show a few example predictions from the final val eval
    print("\nSample validation predictions:")
    for word, gold, pred, is_match in examples:
        status = "✅" if is_match else "❌"
        print(f"  {status} {word:<15} gold={' '.join(gold):<25} pred={' '.join(pred)}")

    # ── Final TEST evaluation -- using the best-on-val checkpoint, exactly
    # once, on data the model and the checkpoint-selection process have
    # both never seen. Do not tune anything based on this number.
    best_ckpt = torch.load(os.path.join(args.checkpoint_dir, "best_model.pt"),
                            map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    test_token_acc, test_word_acc, test_examples = evaluate(model, test_loader, vocabs, device)
    print(f"\nFINAL TEST word-exact-match accuracy: {test_word_acc:.3f}  "
          f"(token_acc={test_token_acc:.3f}, n={len(test_entries)})")
    print("Sample test predictions:")
    for word, gold, pred, is_match in test_examples:
        status = "✅" if is_match else "❌"
        print(f"  {status} {word:<15} gold={' '.join(gold):<25} pred={' '.join(pred)}")


if __name__ == "__main__":
    main()
