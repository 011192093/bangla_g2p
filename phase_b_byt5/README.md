# ByT5 Fine-tuning Experiment for Bangla G2P

A second, independent modeling approach to compare against the GRU
pipeline (`phase_b/`): fine-tuning Google's pretrained ByT5-small model
instead of training a seq2seq model from scratch.

## IMPORTANT — this was NOT fully tested end-to-end

The sandbox used to build this had no access to huggingface.co (only a
restricted domain allowlist), so the actual pretrained ByT5-small
weights could never be downloaded and tested here. What WAS verified:

- The data pipeline (`data.py`) — loading, splitting, formatting —
  confirmed working correctly on real lexicon data
- The training loop mechanics (forward pass, loss, backward pass,
  gradient computation) — confirmed working using a tiny randomly-
  initialized T5 model of the same architecture family
- Script syntax — confirmed valid Python, no syntax errors

What was NOT tested:
- Actually downloading and loading the real byt5-small pretrained weights
- A real training run to convergence
- The `evaluate()` function's `model.generate()` call in practice

**Run this on your own machine with real internet access first with a
tiny lexicon (10-20 words, 1-2 epochs) to confirm it works end-to-end
before trusting it on your full ~15,000-word lexicon.** Treat this as a
solid, logically-verified starting scaffold, not a fully proven pipeline
the way the GRU code was (which WAS fully tested in the sandbox).

## Setup

```powershell
cd phase_b_byt5
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## GPU strongly recommended

ByT5-small is ~300M parameters -- roughly 650x larger than the GRU
model (457K params). This will be very slow on CPU. If you don't have
a local GPU, consider Google Colab (free tier includes GPU access) --
upload this folder, mount your data, and run there instead.

## First run — sanity check with a tiny sample

Before running on your full lexicon, copy a small 20-30 word slice of
your lexicon to `data/lexicon_tiny.tsv` and run:

```powershell
cd src
python train_byt5.py --lexicon ../data/lexicon_tiny.tsv --epochs 2 --batch_size 4
```

This should complete in a few minutes even on CPU and confirms the
pipeline actually works before committing to a long run on your full
lexicon.

## Full run

```powershell
python train_byt5.py --lexicon ../data/lexicon.tsv --epochs 10 --batch_size 8
```

Start with fewer epochs than the GRU used (80) -- ByT5 is pretrained,
so it typically converges much faster. Watch `val_word_acc`: if it's
still climbing at epoch 10, add more; if it plateaus quickly (likely),
stop early to save time.

## Comparing against the GRU

Both pipelines use the identical train/val split logic (same seed, same
ratio) on the same lexicon file, so if you point both at the same
`lexicon.tsv`, they evaluate on the same held-out words -- making the
word-exact-match accuracy numbers directly comparable.

## What to report either way

- If ByT5 beats the GRU: you have a stronger system and a legitimate
  "we tried X and it helped" result for the paper
- If it doesn't: that's equally valid evidence your GRU + data-scaling
  approach was the right call for this lexicon size, worth stating
  explicitly rather than omitting the comparison
