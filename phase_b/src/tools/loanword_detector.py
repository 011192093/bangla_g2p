# -*- coding: utf-8 -*-
"""
Loanword detector for the Bangla G2P lexicon.

Detects English-origin words (or English-derived words embedded in
otherwise-native forms) by their PHONEME string, independent of whatever
register tag they currently carry. It works even on words that are a mix
of a native root + an English loanword segment (e.g. বাসস্ট্যান্ড =
বাস(native "residence" homonym collision aside)+স্ট্যান্ড("stand", English)),
because it scores using a sliding window over phoneme character n-grams
rather than the whole word at once.

Method
------
A multinomial Naive Bayes classifier over character n-grams (n=2..4) of the
phoneme string (tokens joined with no separator, so cluster patterns like
"sT", "spO", "kr" are visible to the model), trained directly from the
existing lexicon's own tag column. No external ML libraries needed.

  classes:
    LOAN    = {loanword, proper_noun}
    NATIVE  = {native, tatsama, tadbhava, deshi}
  (compound/unknown/loanword_nativized are excluded from training since
   they are mixed-origin or ambiguous by definition, but can still be
   scored.)

Two scoring modes:
  1. whole-word score: log P(LOAN | phoneme) - log P(NATIVE | phoneme),
     using all n-grams in the phoneme string.
  2. best-window score: the same log-odds computed on the single highest-
     scoring contiguous substring of the phoneme (default: length 3-8
     characters), which is what actually detects an embedded loanword
     segment inside an otherwise-native compound. This is the score used
     for ranking "native"-tagged candidates, since it catches partial
     contamination (e.g. a native root + English suffix) that a whole-
     word score would dilute.

Usage
-----
Train (writes a model file, run once / whenever the lexicon changes a lot):
    python loanword_detector.py train --lexicon <path> --model <out.json>

Scan a batch file for native-tagged words that look like loanwords:
    python loanword_detector.py scan --model <model.json> --batch <path>
        [--tag native] [--top 500] [--out <report.txt>]

Score a single word's phoneme directly (debugging):
    python loanword_detector.py score --model <model.json> --phoneme "s T a r T"
"""
import argparse
import io
import json
import math
import os
import unicodedata
from collections import Counter, defaultdict

NGRAM_RANGE = (2, 4)
WINDOW_RANGE = (3, 8)  # character lengths for the sliding-window score
LOAN_TAGS = {'loanword', 'proper_noun'}
NATIVE_TAGS = {'native', 'tatsama', 'tadbhava', 'deshi'}


def nfc(s):
    return unicodedata.normalize('NFC', s)


def load_tsv(path):
    raw = io.open(path, encoding='utf-8', newline='').read()
    lines = raw.split('\n')
    if lines and lines[-1] == '':
        lines = lines[:-1]
    rows = []
    for l in lines:
        p = l.split('\t')
        if len(p) != 3:
            continue
        w, ph, t = p
        rows.append((nfc(w), ph.rstrip('\r').strip(), t.rstrip('\r').strip()))
    return rows


def phoneme_key(phoneme):
    """Collapse a space-separated phoneme string into a bare character
    sequence so n-grams can see cross-token clusters (e.g. 's','T' -> 'sT')."""
    return phoneme.replace(' ', '')


def char_ngrams(s, nmin, nmax):
    out = []
    L = len(s)
    for n in range(nmin, nmax + 1):
        for i in range(L - n + 1):
            out.append(s[i:i + n])
    return out


def train(lexicon_path, model_path):
    rows = load_tsv(lexicon_path)
    loan_counts = Counter()
    native_counts = Counter()
    loan_docs = 0
    native_docs = 0
    for w, ph, t in rows:
        if not ph:
            continue
        cls = None
        if t in LOAN_TAGS:
            cls = 'loan'
        elif t in NATIVE_TAGS:
            cls = 'native'
        else:
            continue
        key = phoneme_key(ph)
        grams = set(char_ngrams(key, *NGRAM_RANGE))  # set: presence, not count, per doc
        if cls == 'loan':
            loan_docs += 1
            for g in grams:
                loan_counts[g] += 1
        else:
            native_docs += 1
            for g in grams:
                native_counts[g] += 1

    vocab = set(loan_counts) | set(native_counts)
    V = len(vocab)
    model = {
        'loan_docs': loan_docs,
        'native_docs': native_docs,
        'vocab_size': V,
        'loan_counts': dict(loan_counts),
        'native_counts': dict(native_counts),
        'ngram_range': list(NGRAM_RANGE),
    }
    with io.open(model_path, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False)
    print('trained on %d loan docs, %d native docs, vocab=%d -> %s' % (
        loan_docs, native_docs, V, model_path))
    return model


def load_model(model_path):
    with io.open(model_path, encoding='utf-8') as f:
        return json.load(f)


def _log_odds_for_grams(grams, model):
    """log P(loan)-log P(native) for a bag of n-gram features, using
    Laplace-smoothed presence counts (Bernoulli-ish multinomial NB)."""
    loan_docs = model['loan_docs']
    native_docs = model['native_docs']
    V = model['vocab_size']
    loan_counts = model['loan_counts']
    native_counts = model['native_counts']
    loan_total = sum(loan_counts.values())
    native_total = sum(native_counts.values())

    score = math.log(loan_docs + 1) - math.log(native_docs + 1)  # class prior
    for g in grams:
        lc = loan_counts.get(g, 0)
        nc = native_counts.get(g, 0)
        p_loan = (lc + 1) / (loan_total + V)
        p_native = (nc + 1) / (native_total + V)
        score += math.log(p_loan) - math.log(p_native)
    return score


def score_whole(phoneme, model):
    key = phoneme_key(phoneme)
    grams = char_ngrams(key, *model.get('ngram_range', NGRAM_RANGE))
    if not grams:
        return 0.0
    return _log_odds_for_grams(grams, model) / len(grams)  # normalize by length


def _gram_value(g, model, prior_per_gram):
    """Per-n-gram log P(loan)-log P(native) contribution, plus an even
    share of the class-prior term so window sums stay comparable."""
    loan_counts = model['loan_counts']
    native_counts = model['native_counts']
    V = model['vocab_size']
    loan_total = model['_loan_total']
    native_total = model['_native_total']
    lc = loan_counts.get(g, 0)
    nc = native_counts.get(g, 0)
    p_loan = (lc + 1) / (loan_total + V)
    p_native = (nc + 1) / (native_total + V)
    return math.log(p_loan) - math.log(p_native) + prior_per_gram


def score_best_window(phoneme, model, window_range=WINDOW_RANGE):
    """Find the highest-scoring contiguous substring of the phoneme key
    without recomputing n-grams per window: build one log-odds value per
    (n-gram length, start position) in a single O(L) pass, prefix-sum each
    length's array, then evaluate every window with O(1) range-sum lookups.
    This is what detects a loanword segment embedded in an otherwise-
    native word, at a speed that scales to the whole lexicon."""
    key = phoneme_key(phoneme)
    nmin, nmax = model.get('ngram_range', NGRAM_RANGE)
    L = len(key)
    if L == 0:
        return 0.0, key

    if '_loan_total' not in model:
        model['_loan_total'] = sum(model['loan_counts'].values())
        model['_native_total'] = sum(model['native_counts'].values())

    loan_docs = model['loan_docs']
    native_docs = model['native_docs']
    prior = math.log(loan_docs + 1) - math.log(native_docs + 1)

    # prefix[n][k] = sum of gram_val for n-grams starting at positions 0..k-1
    prefixes = {}
    for n in range(nmin, nmax + 1):
        vals = [0.0] * max(0, L - n + 1)
        for i in range(len(vals)):
            g = key[i:i + n]
            # spread the prior evenly per n-gram-length series so it isn't
            # double counted across n in (2..4); divide once at the end instead
            vals[i] = _gram_value(g, model, 0.0)
        pre = [0.0] * (len(vals) + 1)
        for i, v in enumerate(vals):
            pre[i + 1] = pre[i] + v
        prefixes[n] = pre

    wmin, wmax = window_range
    best_score = None
    best_win = None
    for wlen in range(wmin, min(wmax, L) + 1):
        for i in range(L - wlen + 1):
            total = 0.0
            count = 0
            for n in range(nmin, nmax + 1):
                pre = prefixes[n]
                # n-grams of length n fully inside [i, i+wlen) start at
                # s in [i, i+wlen-n], i.e. pre indices [i, i+wlen-n+1)
                lo = i
                hi = i + wlen - n + 1
                if hi <= 0 or lo >= len(pre) - 1:
                    continue
                lo = max(lo, 0)
                hi = min(hi, len(pre) - 1)
                if hi <= lo:
                    continue
                total += pre[hi] - pre[lo]
                count += hi - lo
            if count == 0:
                continue
            s = total / count + prior
            if best_score is None or s > best_score:
                best_score = s
                best_win = key[i:i + wlen]
    if best_score is None:
        return score_whole(phoneme, model), key
    return best_score, best_win


def cmd_train(args):
    train(args.lexicon, args.model)


def cmd_score(args):
    model = load_model(args.model)
    whole = score_whole(args.phoneme, model)
    best, win = score_best_window(args.phoneme, model)
    print('whole_word_score=%.4f' % whole)
    print('best_window_score=%.4f window=%r' % (best, win))


def cmd_scan(args):
    model = load_model(args.model)
    rows = load_tsv(args.batch)
    target_tags = set(t.strip() for t in args.tag.split(','))
    results = []
    for w, ph, t in rows:
        if t not in target_tags:
            continue
        if not ph:
            continue
        best, win = score_best_window(ph, model)
        results.append((best, w, ph, t, win))
    results.sort(key=lambda r: -r[0])
    top = results[:args.top] if args.top else results

    out_path = args.out or (os.path.splitext(args.batch)[0] + '_loanword_candidates.txt')
    with io.open(out_path, 'w', encoding='utf-8') as f:
        f.write('# %d candidates (of %d %s-tagged words scanned), sorted by loanword-likelihood\n' % (
            len(top), len(results), args.tag))
        f.write('# score\tword\tphoneme\ttag\ttrigger_window\n')
        for best, w, ph, t, win in top:
            f.write('%.4f\t%s\t%s\t%s\t%s\n' % (best, w, ph, t, win))
    print('scanned %d %s-tagged words, wrote top %d candidates -> %s' % (
        len(results), args.tag, len(top), out_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='train a model from a tagged lexicon.tsv')
    t.add_argument('--lexicon', required=True)
    t.add_argument('--model', required=True)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser('score', help='score one phoneme string directly')
    s.add_argument('--model', required=True)
    s.add_argument('--phoneme', required=True)
    s.set_defaults(func=cmd_score)

    sc = sub.add_parser('scan', help='rank a batch file\'s candidates by loanword-likelihood')
    sc.add_argument('--model', required=True)
    sc.add_argument('--batch', required=True)
    sc.add_argument('--tag', default='native', help='comma-separated tag(s) to scan (default: native)')
    sc.add_argument('--top', type=int, default=1000)
    sc.add_argument('--out')
    sc.set_defaults(func=cmd_scan)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
