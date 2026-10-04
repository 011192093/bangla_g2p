"""
Fixes phase_b/src/sustlex_needs_E_e_review.tsv in three passes:

  1. Phonemes: every row gets regenerated from the real rule engine
     (G.text_to_phonemes), replacing whatever naive process produced the
     file originally. Diffing the file against the engine showed the file
     disagreed with the engine on far more than just এ vowel quality --
     schwa deletion and conjunct collapsing were wrong on ~40% of rows
     (e.g. এইকমকে: file "e i k o m k e" vs engine "e i k m k e"; ডব্লিউ
     clusters not collapsed). The engine is the same one behind the
     ~15k already-verified lexicon entries, so this is a mechanical,
     trustworthy fix, not a guess.

  2. Register tags: classify_register() from convert_lexicon.py (the same
     heuristic already used for the rest of the lexicon).

  3. Independent-এ vowel quality (E vs e): the ONE thing the engine can't
     derive (VOWELS hardcodes এ to "e" -- see train_e_vowel_classifier.py
     for why). Predicted via the trained classifier; only applied when
     confidence clears --threshold, else left as the engine default and
     flagged in the review-needed output.

Usage:
    python fix_review_lexicon.py --input ../src/sustlex_needs_E_e_review.tsv \
        --output ../src/sustlex_fixed.tsv \
        --review-out ../src/sustlex_still_needs_review.tsv
"""

import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import g2p_engine as G
from normalizer import normalize
from convert_lexicon import classify_register
from train_e_vowel_classifier import extract_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--review-out", required=True)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "e_vowel_model.pkl"))
    ap.add_argument("--threshold", type=float, default=0.75,
                     help="Classifier confidence needed to auto-apply an E/e prediction")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        saved = pickle.load(f)
    vec, clf = saved["vectorizer"], saved["model"]

    fixed_rows = []
    review_rows = []
    stats = {"total": 0, "phon_changed": 0, "e_word_count": 0,
              "e_high_conf": 0, "e_low_conf": 0, "engine_error": 0}

    with open(args.input, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            word, old_phon = parts[0].strip(), parts[1].strip()
            stats["total"] += 1

            try:
                eng_toks = G.text_to_phonemes(normalize(word))
            except Exception:
                stats["engine_error"] += 1
                fixed_rows.append((word, old_phon, "unknown"))
                review_rows.append((word, old_phon, "engine raised an exception -- needs a human look"))
                continue

            if eng_toks != old_phon.split():
                stats["phon_changed"] += 1

            tag = classify_register(word, " ".join(eng_toks))

            conf_note = None
            if word.startswith("এ") and eng_toks and eng_toks[0] in ("e", "E"):
                stats["e_word_count"] += 1
                feats = extract_features(word, tag)
                X = vec.transform([feats])
                prob_E = clf.predict_proba(X)[0][1]
                predicted = "E" if prob_E >= 0.5 else "e"
                confidence = prob_E if predicted == "E" else 1 - prob_E

                if confidence >= args.threshold:
                    eng_toks[0] = predicted
                    stats["e_high_conf"] += 1
                else:
                    stats["e_low_conf"] += 1
                    conf_note = f"low-confidence এ vowel: predicted {predicted} (p={prob_E:.2f}), left as engine default {eng_toks[0]!r}"

            new_phon = " ".join(eng_toks)
            fixed_rows.append((word, new_phon, tag))
            if conf_note:
                review_rows.append((word, new_phon, conf_note))

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# word\tphonemes\tregister\n")
        for word, phon, tag in fixed_rows:
            f.write(f"{word}\t{phon}\t{tag}\n")

    with open(args.review_out, "w", encoding="utf-8") as f:
        f.write("# word\tphonemes\tnote\n")
        for word, phon, note in review_rows:
            f.write(f"{word}\t{phon}\t{note}\n")

    print(f"Scanned {stats['total']} words")
    print(f"  Phonemes changed by re-running the engine: {stats['phon_changed']}")
    print(f"  Independent-এ words: {stats['e_word_count']}")
    print(f"    auto-applied (confidence >= {args.threshold}): {stats['e_high_conf']}")
    print(f"    flagged for manual review (low confidence): {stats['e_low_conf']}")
    print(f"  Engine errors: {stats['engine_error']}")
    print(f"\nWrote {len(fixed_rows)} rows -> {args.output}")
    print(f"Wrote {len(review_rows)} rows needing a human look -> {args.review_out}")


if __name__ == "__main__":
    main()
