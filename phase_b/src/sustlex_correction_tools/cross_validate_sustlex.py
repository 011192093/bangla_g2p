"""
Cross-validate SUSTlex against your existing lexicon AND your rule
engine, to auto-flag only the words that actually need human review --
instead of checking all 125,000 by hand.

Usage (run from phase_b/src or wherever g2p_engine.py is importable):
    python cross_validate_sustlex.py \
        --sustlex ../../sustlex_converted.tsv \
        --your_lexicon ../data/lexicon.tsv \
        --g2p_src_dir ..\..\bangla_g2p\src \
        --output_agree sustlex_agree.tsv \
        --output_disagree sustlex_disagree.tsv
"""
import sys
import os
import argparse


def load_lexicon(path):
    entries = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2 and parts[0].strip():
                entries[parts[0].strip()] = parts[1].strip()
    return entries


def normalize_for_compare(phon_str):
    """Strip spacing/tokenization differences so we compare CONTENT,
    not formatting -- same lesson learned from the Google-lexicon
    cross-check earlier this session."""
    return phon_str.replace(" ", "").replace("|", "/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sustlex", required=True)
    parser.add_argument("--your_lexicon", required=True)
    parser.add_argument("--g2p_src_dir", required=True)
    parser.add_argument("--output_agree", default="sustlex_agree.tsv")
    parser.add_argument("--output_disagree", default="sustlex_disagree.tsv")
    parser.add_argument("--output_new", default="sustlex_new_words.tsv")
    args = parser.parse_args()

    sys.path.insert(0, args.g2p_src_dir)
    try:
        from g2p_engine import word_to_phonemes  # adjust if your function name differs
        HAS_ENGINE = True
    except ImportError as e:
        print(f"Could not import rule engine: {e}")
        print("Continuing with lexicon-only cross-validation (no rule-engine check).")
        HAS_ENGINE = False

    sustlex = load_lexicon(args.sustlex)
    your_lex = load_lexicon(args.your_lexicon)

    print(f"SUSTlex: {len(sustlex)} words")
    print(f"Your lexicon: {len(your_lex)} words")

    overlap = set(sustlex.keys()) & set(your_lex.keys())
    new_words = set(sustlex.keys()) - set(your_lex.keys())
    print(f"Overlap with your lexicon: {len(overlap)} words")
    print(f"New words (not in your lexicon at all): {len(new_words)} words")

    agree, disagree = [], []
    for word in overlap:
        sust_phon = sustlex[word]
        your_phon = your_lex[word]

        # take first variant if SUSTlex has multiple (pipe-separated)
        sust_variants = [normalize_for_compare(v) for v in sust_phon.split("|")]
        your_norm = normalize_for_compare(your_phon)

        if your_norm in sust_variants:
            agree.append((word, your_phon, sust_phon))
        else:
            disagree.append((word, your_phon, sust_phon))

    print(f"\nAgree with your lexicon: {len(agree)}  ({len(agree)/len(overlap)*100:.1f}%)")
    print(f"Disagree: {len(disagree)}  ({len(disagree)/len(overlap)*100:.1f}%)")

    with open(args.output_agree, 'w', encoding='utf-8') as f:
        f.write("# word\tyour_lexicon\tsustlex\n")
        for word, yp, sp in agree:
            f.write(f"{word}\t{yp}\t{sp}\n")

    with open(args.output_disagree, 'w', encoding='utf-8') as f:
        f.write("# word\tyour_lexicon\tsustlex\tNEEDS_REVIEW\n")
        for word, yp, sp in disagree:
            f.write(f"{word}\t{yp}\t{sp}\t\n")

    with open(args.output_new, 'w', encoding='utf-8') as f:
        f.write("# word\tsustlex_phonemes\tregister\n")
        for word in new_words:
            f.write(f"{word}\t{sustlex[word]}\tunknown\n")

    print(f"\nWritten:")
    print(f"  {args.output_agree}  -- trust these directly, no review needed")
    print(f"  {args.output_disagree}  -- YOUR ACTUAL review list ({len(disagree)} words, not 125,000)")
    print(f"  {args.output_new}  -- brand new words, still need eventual review")

    if HAS_ENGINE:
        print("\n--- Cross-checking NEW words against your rule engine ---")
        rule_agree, rule_disagree = [], []
        for word in list(new_words)[:5000]:  # cap for speed, expand if needed
            try:
                rule_pred = word_to_phonemes(word)  # adjust call signature as needed
                rule_str = normalize_for_compare(" ".join(rule_pred) if isinstance(rule_pred, list) else rule_pred)
            except Exception:
                continue
            sust_variants = [normalize_for_compare(v) for v in sustlex[word].split("|")]
            if rule_str in sust_variants:
                rule_agree.append(word)
            else:
                rule_disagree.append(word)

        total_checked = len(rule_agree) + len(rule_disagree)
        if total_checked:
            print(f"Rule engine vs SUSTlex on {total_checked} new words: "
                  f"{len(rule_agree)} agree ({len(rule_agree)/total_checked*100:.1f}%), "
                  f"{len(rule_disagree)} disagree")


if __name__ == "__main__":
    main()
