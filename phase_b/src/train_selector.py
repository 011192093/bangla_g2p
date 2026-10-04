"""
Rule-vs-Neural Selector
==========================
Learns, from the word's SPELLING ALONE, whether the rule engine or the
neural model is more likely to be correct for that word -- so at
inference time you pick the better prediction per word instead of
always trusting one system.

This is the same technique as the register classifier built earlier in
this project (character n-grams -> a trained classifier), just applied
to a different target: not "what register is this word," but "which
of my two G2P systems should I trust for this word."

INPUT FORMAT (tab-separated, one row per held-out word):
    word<TAB>gold_phonemes<TAB>rule_prediction<TAB>neural_prediction

You need to generate this file from your own compare_with_engine.py --
see generate_comparison_file.py for a template showing exactly what
that script needs to dump. The column format matches what
compare_with_engine.py already computes internally (it already knows
gold, rule's guess, and neural's guess for every held-out word; it
just currently only prints aggregate counts, not the per-word detail).

USAGE:
    python train_selector.py comparison_data.tsv

OUTPUT:
    - Honest, HELD-OUT ensemble accuracy (not just a theoretical ceiling)
    - selector.pkl -- the trained model, reusable at inference time
    - A clear comparison: rule alone / neural alone / this selector /
      theoretical perfect-selector ceiling
"""
import sys
import pickle
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_comparison(path):
    """Returns list of (word, gold, rule_pred, neural_pred)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\r\n")
            if not raw.strip() or raw.startswith("#"):
                continue
            parts = raw.split("\t")
            if len(parts) < 4:
                continue
            word, gold, rule_pred, neural_pred = parts[0], parts[1], parts[2], parts[3]
            rows.append((word, gold, rule_pred, neural_pred))
    return rows


def build_labels(rows):
    """
    Label = 1 means 'trust the rule engine for this word', assigned
    ONLY to words where the rule is correct and neural is NOT -- these
    are exactly the words a selector needs to catch, since defaulting
    to neural already handles every other case correctly or incorrectly
    the same way regardless of the selector.
    """
    words, labels = [], []
    for word, gold, rule_pred, neural_pred in rows:
        rule_ok = (rule_pred.strip() == gold.strip())
        neural_ok = (neural_pred.strip() == gold.strip())
        trust_rule = rule_ok and not neural_ok
        words.append(word)
        labels.append(1 if trust_rule else 0)
    return words, np.array(labels)


def evaluate_ensemble(rows, trust_rule_predictions):
    """
    Given a boolean array (one per row) saying whether the selector
    chose to trust the rule engine for that word, compute the
    resulting ensemble's accuracy -- this is the real, honest number,
    not the theoretical ceiling.
    """
    correct = 0
    for (word, gold, rule_pred, neural_pred), trust_rule in zip(rows, trust_rule_predictions):
        chosen = rule_pred if trust_rule else neural_pred
        if chosen.strip() == gold.strip():
            correct += 1
    return correct / len(rows)


def main():
    if len(sys.argv) != 2:
        print("Usage: python train_selector.py comparison_data.tsv")
        sys.exit(1)

    rows = load_comparison(sys.argv[1])
    if not rows:
        print("No valid rows found. Check the file format: "
              "word<TAB>gold<TAB>rule_pred<TAB>neural_pred")
        sys.exit(1)

    print(f"Loaded {len(rows)} comparison rows")

    # baseline numbers, for context
    rule_acc = sum(1 for w,g,r,n in rows if r.strip()==g.strip()) / len(rows)
    neural_acc = sum(1 for w,g,r,n in rows if n.strip()==g.strip()) / len(rows)
    ceiling = sum(1 for w,g,r,n in rows if r.strip()==g.strip() or n.strip()==g.strip()) / len(rows)
    print(f"Rule alone:              {rule_acc*100:.2f}%")
    print(f"Neural alone:            {neural_acc*100:.2f}%")
    print(f"Theoretical ceiling:     {ceiling*100:.2f}%  (perfect selector, upper bound)")
    print()

    words, labels = build_labels(rows)
    print(f"Words where ONLY rule is right (what the selector must catch): "
          f"{labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")

    # train/test split -- the selector's own accuracy must be measured
    # on words it never trained on, same discipline as every other
    # classifier in this project
    idx = np.arange(len(rows))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42,
                                            stratify=labels if labels.sum() >= 5 else None)

    words_train = [words[i] for i in idx_train]
    words_test  = [words[i] for i in idx_test]
    y_train = labels[idx_train]
    y_test  = labels[idx_test]
    rows_test = [rows[i] for i in idx_test]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
    X_train = vectorizer.fit_transform(words_train)
    X_test = vectorizer.transform(words_test)

    if y_train.sum() < 3:
        print("\nToo few 'trust rule' examples to train a classifier reliably "
              "(need at least a handful). With this little signal, defaulting "
              "to 'always trust neural' is the safer choice -- see the "
              "'always neural' row below.")
        clf = None
    else:
        base = LinearSVC(class_weight="balanced", C=0.5, max_iter=20000)
        clf = CalibratedClassifierCV(base, cv=min(3, int(y_train.sum())))
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("\nHeld-out classification report (predicting 'trust rule for this word'):")
        print(classification_report(y_test, y_pred, zero_division=0))

    print("=" * 60)
    print("FINAL COMPARISON -- all numbers below are on the SAME held-out test rows")
    print("=" * 60)

    test_rule_acc = sum(1 for w,g,r,n in rows_test if r.strip()==g.strip()) / len(rows_test)
    test_neural_acc = sum(1 for w,g,r,n in rows_test if n.strip()==g.strip()) / len(rows_test)
    test_ceiling = sum(1 for w,g,r,n in rows_test
                        if r.strip()==g.strip() or n.strip()==g.strip()) / len(rows_test)

    always_neural = [False] * len(rows_test)
    always_rule   = [True] * len(rows_test)
    print(f"Rule alone (this split):     {test_rule_acc*100:.2f}%")
    print(f"Neural alone (this split):   {test_neural_acc*100:.2f}%")
    print(f"Always trust neural:         {evaluate_ensemble(rows_test, always_neural)*100:.2f}%  "
          f"(same as 'neural alone' above, shown for comparison)")
    print(f"Always trust rule:           {evaluate_ensemble(rows_test, always_rule)*100:.2f}%  "
          f"(same as 'rule alone' above, shown for comparison)")
    print(f"Theoretical ceiling (this split): {test_ceiling*100:.2f}%  "
          f"(upper bound -- no selector can beat this, on these rows)")

    if clf is not None:
        learned_choice = [bool(p) for p in y_pred]
        learned_acc = evaluate_ensemble(rows_test, learned_choice)
        print(f"Learned selector (this script):   {learned_acc*100:.2f}%")

        with open("selector.pkl", "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "clf": clf}, f)
        print("\nSaved selector.pkl -- see predict_with_selector.py for how to use it")
    else:
        print("(no learned selector trained -- insufficient positive examples)")


if __name__ == "__main__":
    main()
