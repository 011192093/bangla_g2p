# Bangla G2P Module

Rule-based (Phase A) + neural exception (Phase B) grapheme-to-phoneme
converter for Bangla, built as the foundation for a Bangla voice cloning
system.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project structure

```
bangla_g2p/
├── src/
│   ├── phoneme_inventory.py   # phoneme set + Unicode char classification
│   └── g2p_engine.py          # core rule-based segmentation + conversion
├── tests/
│   └── test_g2p_engine.py     # regression tests — add one per bug found
├── data/
│   └── lexicon_seed.tsv       # exception word list (grows over time)
├── requirements.txt
└── README.md
```

## Running it

```bash
# Quick manual test
python3 src/g2p_engine.py

# Run the test suite
python3 -m pytest tests/ -v

# Use it in your own script
python3 -c "
import sys; sys.path.insert(0, 'src')
from g2p_engine import text_to_phonemes
print(text_to_phonemes('আমি বাংলায় গান গাই'))
"
```

## Workflow for improving accuracy

1. Pick a batch of words/sentences (start with common vocabulary).
2. Run them through `text_to_phonemes`.
3. Have a native Bangla speaker mark which outputs sound wrong.
4. For each wrong case:
   - If it's a **systematic rule bug** (e.g. wrong inherent-vowel deletion
     pattern) -> fix the rule in `g2p_engine.py`, add a regression test in
     `tests/test_g2p_engine.py`.
   - If it's a **one-off irregular word/loanword** -> add the correct
     transcription to `data/lexicon_seed.tsv`.
5. Re-run tests, commit, repeat.
6. Once `lexicon_seed.tsv` has 2,000+ verified entries, move to Phase B
   (train a small neural model to catch words the rules can't handle).

## Git setup

```bash
git init
git add .
git commit -m "Initial Bangla G2P rule-based engine"
```

Push to GitHub for collaboration / to cite as your open-source artifact
in the paper.
