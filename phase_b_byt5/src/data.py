"""
ByT5 Fine-tuning -- Data Pipeline
===================================
Reuses the exact same lexicon.tsv format as the GRU pipeline
(word<TAB>phonemes<TAB>register), so both experiments train/evaluate
on identical data for a fair comparison.

ByT5 needs no custom vocabulary -- it operates on raw UTF-8 bytes via
its own tokenizer, so Bangla script (including any conjunct/nukta
representation) is handled natively without the segmentation logic
your rule engine and GRU pipeline needed to build by hand.

We format each example as:
    input:  "bn2phoneme: <register>: <word>"
    target: "<phonemes space-separated>"

Including the register tag directly in the input text (rather than as
a separate embedding, as in the GRU model) is the standard way to
inject side-information into a text-to-text model like T5/ByT5.
"""

import random


def load_lexicon(path):
    """Returns list of (word, phoneme_list, register) tuples."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word = parts[0].strip()
            phonemes = parts[1].strip().split()
            register = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else "unknown"
            entries.append((word, phonemes, register))
    return entries


def format_example(word, register):
    """Input text format fed to ByT5."""
    return f"bn2phoneme: {register}: {word}"


def format_target(phonemes):
    """Target text format -- space-separated phonemes, same as the
    GRU pipeline's token scheme, so outputs are directly comparable."""
    return " ".join(phonemes)


def train_val_split(entries, val_ratio=0.15, seed=42):
    """Identical split logic to the GRU pipeline (same seed) so both
    experiments can be evaluated on the SAME held-out words if the
    same lexicon file is used -- enabling a fair, direct comparison."""
    rng = random.Random(seed)
    shuffled = entries[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def build_hf_dataset(entries):
    """Convert entries into the dict-of-lists format Hugging Face
    datasets/Trainer expects."""
    inputs = [format_example(w, r) for w, p, r in entries]
    targets = [format_target(p) for w, p, r in entries]
    words = [w for w, p, r in entries]
    return {"input_text": inputs, "target_text": targets, "word": words}
