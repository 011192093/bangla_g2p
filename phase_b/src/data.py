"""
Phase B — Data Pipeline
========================
Loads the lexicon (word, phonemes, register_tag), builds character-level
vocabularies for input (Bangla graphemes) and output (phonemes), and
provides a PyTorch Dataset + collate function for training.

Expected lexicon format (tab-separated), one entry per line:
    word<TAB>phoneme1 phoneme2 phoneme3 ...<TAB>register_tag

register_tag is optional — if missing, defaults to "unknown".
Comment lines starting with # are skipped.

Usage:
    from data import load_lexicon, build_vocabs, G2PDataset
"""

import os
import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]

# Known register tags — used as an auxiliary input feature to the model.
REGISTER_TAGS = [
    "unknown", "native", "tatsama", "tadbhava", "deshi",
    "loanword", "loanword_nativized", "compound", "proper_noun",
]


def load_lexicon(path: str) -> List[Tuple[str, List[str], str]]:
    """
    Read the TSV lexicon file.
    Returns list of (word, phoneme_list, register_tag).
    """
    entries = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word = parts[0].strip()
            phonemes = parts[1].strip().split()
            register = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else "unknown"
            if register not in REGISTER_TAGS:
                register = "unknown"
            entries.append((word, phonemes, register))
    return entries


def build_vocabs(entries: List[Tuple[str, List[str], str]]):
    """
    Build char-level vocab for Bangla input graphemes and a token-level
    vocab for output phonemes (phonemes are multi-char tokens like "kh",
    "chh", "ng", so they must be treated as whole tokens, not chars).
    """
    input_chars = set()
    output_tokens = set()

    for word, phonemes, _ in entries:
        input_chars.update(list(word))
        output_tokens.update(phonemes)

    input_vocab = SPECIAL_TOKENS + sorted(input_chars)
    output_vocab = SPECIAL_TOKENS + sorted(output_tokens)

    input_stoi = {ch: i for i, ch in enumerate(input_vocab)}
    output_stoi = {tok: i for i, tok in enumerate(output_vocab)}

    register_stoi = {tag: i for i, tag in enumerate(REGISTER_TAGS)}

    return {
        "input_vocab": input_vocab,
        "input_stoi": input_stoi,
        "output_vocab": output_vocab,
        "output_stoi": output_stoi,
        "register_stoi": register_stoi,
    }


def encode_word(word: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi.get(ch, stoi[UNK]) for ch in word]


def encode_phonemes(phonemes: List[str], stoi: Dict[str, int]) -> List[int]:
    ids = [stoi[SOS]]
    ids += [stoi.get(p, stoi[UNK]) for p in phonemes]
    ids += [stoi[EOS]]
    return ids


class G2PDataset(Dataset):
    """
    Each item: (input_ids, output_ids, register_id)
    input_ids:  character ids of the Bangla word
    output_ids: token ids of the phoneme sequence, wrapped with SOS/EOS
    register_id: integer id of the register tag (auxiliary feature)
    """

    def __init__(self, entries, vocabs):
        self.entries = entries
        self.vocabs = vocabs

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        word, phonemes, register = self.entries[idx]
        input_ids = encode_word(word, self.vocabs["input_stoi"])
        output_ids = encode_phonemes(phonemes, self.vocabs["output_stoi"])
        register_id = self.vocabs["register_stoi"].get(register, 0)
        return {
            "word": word,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "output_ids": torch.tensor(output_ids, dtype=torch.long),
            "register_id": torch.tensor(register_id, dtype=torch.long),
        }


def collate_batch(batch, pad_id_in: int, pad_id_out: int):
    """
    Pads a batch of variable-length sequences to the max length in the batch.
    Returns tensors shaped (batch, seq_len).
    """
    words = [b["word"] for b in batch]
    input_lens = [len(b["input_ids"]) for b in batch]
    output_lens = [len(b["output_ids"]) for b in batch]

    max_in = max(input_lens)
    max_out = max(output_lens)

    input_batch = torch.full((len(batch), max_in), pad_id_in, dtype=torch.long)
    output_batch = torch.full((len(batch), max_out), pad_id_out, dtype=torch.long)
    register_batch = torch.zeros(len(batch), dtype=torch.long)

    for i, b in enumerate(batch):
        input_batch[i, :len(b["input_ids"])] = b["input_ids"]
        output_batch[i, :len(b["output_ids"])] = b["output_ids"]
        register_batch[i] = b["register_id"]

    return {
        "words": words,
        "input_ids": input_batch,
        "input_lens": torch.tensor(input_lens, dtype=torch.long),
        "output_ids": output_batch,
        "output_lens": torch.tensor(output_lens, dtype=torch.long),
        "register_ids": register_batch,
    }


def train_val_split(entries, val_ratio=0.15, seed=42):
    """
    Random split, held-out val set never seen during training.
    This is what measures TRUE generalization, not memorization.
    """
    rng = random.Random(seed)
    shuffled = entries[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val
