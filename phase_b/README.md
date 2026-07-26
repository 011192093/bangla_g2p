# Phase B — Neural Exception Model

A character-level seq2seq model (BiGRU encoder + attention decoder) that
learns the phoneme patterns your Phase A rule engine could not derive —
especially loanword স (s vs sh), vowel-height harmony (o vs O) in lexically
conditioned positions, and eventually compound-word behavior once enough
compound examples are in the lexicon.

The decoder is conditioned on a **register tag** (native / tatsama /
loanword / loanword_nativized / compound / etc.) fed in at every decode
step. This directly encodes the finding from Phase A: স→s/sh and o/O are
not recoverable from spelling alone but correlate strongly with
etymological register.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install torch
```

(If you have a CUDA GPU, install the CUDA build of torch from
pytorch.org for much faster training — CPU works fine for a
~2,000-word dataset, just slower.)

## Project structure

```
phase_b/
├── data/
│   ├── lexicon.tsv          ← PUT YOUR REAL 2000-WORD LEXICON HERE
│   └── lexicon_test.tsv     ← small synthetic file, for pipeline testing only
├── src/
│   ├── data.py              ← vocab building, Dataset, collate fn
│   ├── model.py             ← Seq2SeqG2P model (encoder/decoder/attention)
│   ├── train.py             ← training loop, checkpointing, validation
│   ├── predict.py           ← inference on new words, interactive mode
│   └── compare_with_engine.py ← Phase A vs Phase B side-by-side eval
├── checkpoints/
│   └── best_model.pt        ← saved after training (highest val word-accuracy)
└── README.md
```

## Step 1 — Prepare your real lexicon

Convert your `lexicon_seed.tsv` into this format (tab-separated):

```
word<TAB>phoneme1 phoneme2 phoneme3 ...<TAB>register_tag
```

Register tag is one of: `native`, `tatsama`, `tadbhava`, `deshi`,
`loanword`, `loanword_nativized`, `compound`, or `unknown` if not yet
tagged (the model still works without tags, it just loses that signal).

Example:
```
গরম	g o r O m	native
সাইকেল	s a i k e l	loanword
সাহেব	sh a h e b	loanword_nativized
```

Save this as `data/lexicon.tsv`.

If your current lexicon file doesn't have the register column yet, you
can add it as a third tab-separated field, or leave it off — the loader
defaults missing tags to `unknown`.

## Step 2 — Train

```powershell
cd src
python train.py --lexicon ../data/lexicon.tsv --epochs 80 --batch_size 32
```

Key flags:
- `--val_ratio 0.15` — 15% of words held out, NEVER seen during training.
  This is what measures true generalization, not memorization.
- `--epochs 80` — increase if loss is still dropping at the end; decrease
  if validation accuracy plateaus early and starts overfitting (val loss
  rising while train loss keeps falling).
- `--hidden_dim 128 --emb_dim 64` — reasonable defaults for ~2000 words.
  Bump to 256/96 if you grow well past 5,000 words.

Watch for:
```
val_word_acc=0.XX
```
This is the metric that matters — exact phoneme-sequence match on words
the model never trained on. Compare this against your Phase A engine's
accuracy on the same held-out words (see Step 4).

The best checkpoint (by held-out word accuracy) is saved automatically
to `checkpoints/best_model.pt`.

## Step 3 — Try it interactively

```powershell
python predict.py --checkpoint ../checkpoints/best_model.pt --interactive
```

Type any Bangla word, optionally with a register tag:
```
সাইকেল|loanword
মন্দির|tatsama
```

Try words NOT in your lexicon — this tests real generalization, e.g. a
loanword the model has never seen, to see if it learned "loanword shape
→ স gives s" rather than memorizing specific words.

## Step 4 — Compare against your Phase A rule engine

```powershell
python compare_with_engine.py \
    --checkpoint ../checkpoints/best_model.pt \
    --lexicon ../data/lexicon.tsv \
    --g2p_src_dir C:\path\to\bangla_g2p\src
```

This runs both systems on the same held-out words and prints:
```
Phase A rule engine accuracy:   XX%
Phase B neural model accuracy:  YY%
Both correct / rule-only / neural-only / both-wrong breakdown
```

This is your paper's Phase B results table.

## What "good" looks like

With ~2,000 words (85% train / 15% val ≈ 1,700 train / 300 val):

- Token accuracy should climb well past 90% within 40-60 epochs.
- Word-exact-match is a harder, stricter metric — even 60-75% on
  TRULY HELD-OUT words would already beat what pure rules achieve on
  the lexically-conditioned categories (loanword s/sh, o/O exceptions).
- If word accuracy is much lower, likely causes: too few epochs, lexicon
  too small still, or the register tags aren't populated (falls back to
  `unknown` for everyone, losing the strongest signal).

## Extending later

- **Grow the lexicon** — more data is the single highest-leverage
  improvement available right now.
- **Beam search** instead of greedy decode in `predict.py` — usually a
  small but real accuracy bump, worth adding once the base pipeline is
  validated.
- **Pretrain on the rule engine's output**, then fine-tune on the
  lexicon — lets the model start from "phonologically plausible" instead
  of random weights, which can help a lot with this little data.
- **Character-level input augmentation** (feeding normalized text instead
  of raw) — reuse your Phase A `normalizer.py` as a preprocessing step
  before this model, exactly as you do for the rule engine.
