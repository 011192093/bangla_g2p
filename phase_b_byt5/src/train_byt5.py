"""
ByT5 Fine-tuning for Bangla G2P
=================================
Fine-tunes a pretrained ByT5-small model on your lexicon, evaluated with
the exact same word-exact-match methodology as the GRU pipeline, so
results are directly comparable.

Usage:
    python train_byt5.py --lexicon ../data/lexicon.tsv --epochs 10

NOTE ON EPOCHS: ByT5 starts pretrained, so it typically needs FAR fewer
epochs than a from-scratch model like the GRU (which used 80). Start
with 5-10 and watch val_word_acc -- if it's still climbing, add more;
if it plateaus quickly (likely, given pretraining), stop early.

NOTE ON COMPUTE: ByT5-small is ~300M parameters, roughly 650x more than
the GRU's 457K. This WILL be slow on CPU. Strongly recommended to run
this on a GPU if at all possible -- see the note in README.md about
Google Colab as a free option if you don't have a local GPU.
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from data import load_lexicon, train_val_split, format_example, format_target


class G2PByT5Dataset(Dataset):
    def __init__(self, entries, tokenizer, max_input_len=32, max_target_len=32):
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        word, phonemes, register = self.entries[idx]
        input_text = format_example(word, register)
        target_text = format_target(phonemes)

        input_enc = self.tokenizer(
            input_text, max_length=self.max_input_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target_text, max_length=self.max_target_len,
            truncation=True, padding="max_length", return_tensors="pt"
        )

        labels = target_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore pad in loss

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
            "word": word,
            "gold_phonemes": phonemes,
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "words": [b["word"] for b in batch],
        "gold_phonemes": [b["gold_phonemes"] for b in batch],
    }


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, max_gen_len=32):
    model.eval()
    correct_words = 0
    total_words = 0
    examples = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generated = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_length=max_gen_len
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for i, pred_text in enumerate(decoded):
            gold_phonemes = batch["gold_phonemes"][i]
            gold_text = " ".join(gold_phonemes)
            pred_text_norm = pred_text.strip()
            is_match = pred_text_norm == gold_text

            total_words += 1
            if is_match:
                correct_words += 1

            if len(examples) < 10:
                examples.append((batch["words"][i], gold_text, pred_text_norm, is_match))

    word_acc = correct_words / max(1, total_words)
    return word_acc, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", type=str, default="../data/lexicon.tsv")
    parser.add_argument("--model_name", type=str, default="google/byt5-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints")
    parser.add_argument("--eval_every", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: running ByT5-small on CPU will be slow. "
              "Strongly recommend GPU (local or Colab) for this experiment.")

    torch.manual_seed(args.seed)

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    entries = load_lexicon(args.lexicon)
    print(f"Loaded {len(entries)} lexicon entries")

    train_entries, val_entries = train_val_split(entries, args.val_ratio, args.seed)
    print(f"Train: {len(train_entries)}   Val (held-out): {len(val_entries)}")

    train_ds = G2PByT5Dataset(train_entries, tokenizer)
    val_ds = G2PByT5Dataset(val_entries, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_word_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch:3d}  loss={avg_loss:.4f}", end="")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            word_acc, examples = evaluate(model, val_loader, tokenizer, device)
            print(f"  val_word_acc={word_acc:.3f}")

            if word_acc > best_word_acc:
                best_word_acc = word_acc
                model.save_pretrained(os.path.join(args.checkpoint_dir, "byt5_best"))
                tokenizer.save_pretrained(os.path.join(args.checkpoint_dir, "byt5_best"))
        else:
            print()

    print(f"\nBest held-out word-exact-match accuracy: {best_word_acc:.3f}")
    print(f"Checkpoint saved to {args.checkpoint_dir}/byt5_best")

    print("\nSample validation predictions:")
    for word, gold, pred, is_match in examples:
        status = "PASS" if is_match else "FAIL"
        print(f"  {status}  {word:<15} gold={gold:<25} pred={pred}")


if __name__ == "__main__":
    main()
