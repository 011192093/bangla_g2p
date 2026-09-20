"""
Independent-এ vowel-quality classifier
========================================
The rule engine hardcodes the standalone vowel এ to always render as closed
"e" (see VOWELS in phoneme_inventory.py) -- unlike the dependent ে vowel
sign, which has a real context rule (_eekar_is_open in g2p_engine.py).
Checking the ~274 hand-verified এ-initial words already in lexicon.tsv shows
this genuinely isn't spelling-derivable: এখন ("Ekhon", open) and এবং
("ebong", closed) have identical syllable structure but opposite vowel
quality -- these look lexically memorized, not rule-governed.

This trains a small logistic regression on structural features (mirroring
the features _eekar_is_open uses for the dependent-vowel case, since those
are the only linguistically motivated signals available, even though they
don't form a hard rule here) plus the register tag, to predict open (E) vs
closed (e) for new এ-initial words that have no hand verification yet.

Usage:
    python train_e_vowel_classifier.py --lexicon ../data/lexicon.tsv
"""

import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import g2p_engine as G
from normalizer import normalize
from phoneme_inventory import SONORANT_CONSONANTS

REGISTERS = ["native", "loanword", "loanword_nativized", "tatsama", "proper_noun", "compound", "unknown"]


def extract_features(word: str, register: str):
    clusters = G.segment_clusters(normalize(word))
    # clusters[0] is the এ standalone-vowel cluster itself
    next_c = clusters[1] if len(clusters) > 1 else None

    is_conjunct = 0
    has_vowel = 0
    is_word_final = 0
    is_sonorant = 0
    vowel_is_aa_o = 0

    if next_c is not None:
        is_conjunct = int(len(next_c.consonants) > 1)
        has_vowel = int(next_c.vowel_sign is not None and next_c.vowel_sign != "__NONE__")
        is_word_final = int(len(clusters) == 2)
        if next_c.consonants and next_c.consonants[-1] in SONORANT_CONSONANTS:
            is_sonorant = 1
        if next_c.vowel_sign in ("া", "ো"):
            vowel_is_aa_o = 1

    feats = {
        "num_clusters": len(clusters),
        "word_len_chars": len(word),
        "is_conjunct": is_conjunct,
        "has_vowel": has_vowel,
        "is_word_final": is_word_final,
        "is_sonorant": is_sonorant,
        "vowel_is_aa_o": vowel_is_aa_o,
        "starts_with_oi": int(word[:2] == "এই"),
    }
    for r in REGISTERS:
        feats[f"reg_{r}"] = int(register == r)
    return feats


def load_training_data(lexicon_path):
    X_raw, y, words = [], [], []
    with open(lexicon_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, phon, tag = parts
            if not word.startswith("এ") or "|" in phon:
                continue
            toks = phon.split()
            if not toks or toks[0] not in ("e", "E"):
                continue
            X_raw.append(extract_features(word, tag))
            y.append(1 if toks[0] == "E" else 0)
            words.append(word)
    return X_raw, y, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--model-out", default=os.path.join(os.path.dirname(__file__), "e_vowel_model.pkl"))
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    X_raw, y, words = load_training_data(args.lexicon)
    print(f"Training examples: {len(X_raw)}  (E={sum(y)}, e={len(y) - sum(y)})")
    baseline = max(sum(y), len(y) - sum(y)) / len(y)
    print(f"Majority-class baseline accuracy: {baseline:.3f}")

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(X_raw)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(f"5-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})  per-fold: {[round(s,3) for s in scores]}")

    clf.fit(X, y)
    with open(args.model_out, "wb") as f:
        pickle.dump({"vectorizer": vec, "model": clf}, f)
    print(f"Saved model -> {args.model_out}")

    coefs = sorted(zip(vec.get_feature_names_out(), clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    print("\nFeature weights (positive -> pushes toward open E):")
    for name, w in coefs:
        print(f"  {w:+.3f}  {name}")


if __name__ == "__main__":
    main()
