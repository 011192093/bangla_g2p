"""
Hybrid rule+neural router: build, tune, and honestly report.

Protocol (do not deviate):
  1. Run rule engine + neural model on VAL (lexicon_split_val.tsv).
     Compute the oracle ceiling: how many words rule gets right that
     neural gets wrong. That's the absolute cap on any router's gain.
  2. Pick the confidence threshold by scanning VAL accuracy only.
     This is ordinary hyperparameter tuning -- the threshold search uses
     gold labels, but the resulting POLICY (if confidence < t, use rule;
     else use neural) only needs confidence at decision time, so it's
     deployable. The "winner" column (which method matches gold) is a
     separate, oracle-only diagnostic -- it is never used to pick the
     threshold and is not deployable itself.
  3. Freeze that threshold. Apply it ONCE to the FRESH val set
     (lexicon_fresh_val.tsv) and report that number as the real result.
     Never re-tune after seeing the fresh-val number.

Usage:
    python hybrid_analysis.py --checkpoint ../checkpoints/best_model.pt \
        --val_file ../data/lexicon_split_val.tsv \
        --fresh_val_file ../data/lexicon_fresh_val.tsv
"""
import os
import sys
import argparse
import csv
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from data import load_lexicon, encode_word, PAD, SOS, EOS
from model import Seq2SeqG2P
import g2p_engine as G
from normalizer import normalize


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocabs = ckpt["vocabs"]
    args = ckpt["args"]
    pad_id_in = vocabs["input_stoi"][PAD]
    pad_id_out = vocabs["output_stoi"][PAD]
    model = Seq2SeqG2P(
        input_vocab_size=len(vocabs["input_vocab"]),
        output_vocab_size=len(vocabs["output_vocab"]),
        emb_dim=args["emb_dim"],
        hidden_dim=args["hidden_dim"],
        register_vocab_size=len(vocabs["register_stoi"]),
        pad_id_in=pad_id_in,
        pad_id_out=pad_id_out,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocabs


def run_both_systems(entries, model, vocabs, device):
    """Returns list of dicts: word, gold, rule_pred, neural_pred, confidence, winner."""
    input_stoi = vocabs["input_stoi"]
    output_itos = {i: t for t, i in vocabs["output_stoi"].items()}
    register_stoi = vocabs["register_stoi"]
    sos_id = vocabs["output_stoi"][SOS]
    eos_id = vocabs["output_stoi"][EOS]

    rows = []
    for word, gold_phonemes, register in entries:
        # normalize() composes decomposed য়/ড়/ঢ় (among other things) --
        # feed both systems the same normalized text so a word that was
        # never seen in that exact byte form isn't silently garbled for
        # one system but not the other.
        word_norm = normalize(word)
        try:
            rule_pred = G.text_to_phonemes(word_norm)
        except Exception:
            rule_pred = []

        input_ids = torch.tensor([encode_word(word_norm, input_stoi)], dtype=torch.long).to(device)
        input_lens = torch.tensor([len(word_norm)], dtype=torch.long)
        register_id = torch.tensor([register_stoi.get(register, 0)], dtype=torch.long).to(device)
        with torch.no_grad():
            preds, confidences = model.greedy_decode(input_ids, input_lens, register_id, sos_id, eos_id)
        neural_pred = [output_itos.get(t, "?") for t in preds[0]]
        confidence = confidences[0]

        rule_ok = rule_pred == gold_phonemes
        neural_ok = neural_pred == gold_phonemes
        if rule_ok and neural_ok:
            winner = "Both"
        elif rule_ok:
            winner = "Rule"
        elif neural_ok:
            winner = "Neural"
        else:
            winner = "Neither"

        rows.append({
            "word": word, "gold": gold_phonemes, "rule_pred": rule_pred,
            "neural_pred": neural_pred, "confidence": confidence, "winner": winner,
        })
    return rows


def policy_accuracy(rows, threshold):
    """if confidence < threshold: use rule_pred, else neural_pred."""
    correct = 0
    for r in rows:
        chosen = r["rule_pred"] if r["confidence"] < threshold else r["neural_pred"]
        if chosen == r["gold"]:
            correct += 1
    return correct / max(1, len(rows))


def report_summary(rows, label):
    n = len(rows)
    rule_acc = sum(r["rule_pred"] == r["gold"] for r in rows) / n
    neural_acc = sum(r["neural_pred"] == r["gold"] for r in rows) / n
    rule_right_neural_wrong = sum(r["winner"] == "Rule" for r in rows)
    neural_right_rule_wrong = sum(r["winner"] == "Neural" for r in rows)
    neither = sum(r["winner"] == "Neither" for r in rows)
    oracle_acc = sum(r["winner"] in ("Rule", "Neural", "Both") for r in rows) / n

    print(f"\n=== {label} (n={n}) ===")
    print(f"  Rule-only accuracy:    {rule_acc:.4f}")
    print(f"  Neural-only accuracy: {neural_acc:.4f}")
    print(f"  Oracle (perfect router) accuracy: {oracle_acc:.4f}")
    print(f"  Rule right / Neural wrong: {rule_right_neural_wrong} ({100*rule_right_neural_wrong/n:.2f}%)"
          f"  <- ceiling on router gain over neural-only")
    print(f"  Neural right / Rule wrong: {neural_right_rule_wrong} ({100*neural_right_rule_wrong/n:.2f}%)")
    print(f"  Both wrong (neither can help): {neither} ({100*neither/n:.2f}%)")
    return rule_acc, neural_acc, oracle_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="../checkpoints/best_model.pt")
    ap.add_argument("--val_file", default="../data/lexicon_split_val.tsv")
    ap.add_argument("--fresh_val_file", default="../data/lexicon_fresh_val.tsv")
    ap.add_argument("--dump_val_csv", default="../data/hybrid_val_dump.tsv")
    ap.add_argument("--dump_fresh_csv", default="../data/hybrid_fresh_val_dump.tsv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabs = load_model(args.checkpoint, device)

    # ── Step 1+2: everything happens on VAL. Threshold is chosen here. ──
    val_entries = load_lexicon(args.val_file)
    print(f"Loaded {len(val_entries)} VAL words from {args.val_file}")
    val_rows = run_both_systems(val_entries, model, vocabs, device)

    rule_acc, neural_acc, oracle_acc = report_summary(val_rows, "VAL (tuning set)")

    print(f"\n  {'threshold':>10}{'hybrid_acc':>12}{'vs neural-only':>16}")
    best_t, best_acc = None, neural_acc
    for t100 in range(0, 101, 2):
        t = t100 / 100.0
        acc = policy_accuracy(val_rows, t)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_t = t
            marker = "  <- best so far"
        if t100 % 10 == 0:
            print(f"  {t:>10.2f}{acc:>12.4f}{acc - neural_acc:>+16.4f}{marker}")

    if best_t is None:
        print("\nNo threshold beats neural-only accuracy on VAL. "
              "The hybrid does not help here -- recommend NOT deploying it.")
        best_t = 1.0  # degenerate: always use neural (threshold=1.0 means confidence<1.0 always true... )
    else:
        print(f"\nBest threshold on VAL: {best_t:.2f}  "
              f"(hybrid_acc={best_acc:.4f}, neural_only={neural_acc:.4f}, "
              f"gain={best_acc - neural_acc:+.4f})")
        print(f"Oracle ceiling was {oracle_acc:.4f} (gain vs neural-only: {oracle_acc - neural_acc:+.4f})")

    with open(args.dump_val_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["word", "gold", "rule_pred", "neural_pred", "confidence", "winner"])
        for r in val_rows:
            w.writerow([r["word"], " ".join(r["gold"]), " ".join(r["rule_pred"]),
                        " ".join(r["neural_pred"]), f"{r['confidence']:.4f}", r["winner"]])
    print(f"Dumped VAL rows to {args.dump_val_csv}")

    # ── Step 3: freeze best_t, apply ONCE to fresh val, report and stop. ──
    if not os.path.exists(args.fresh_val_file):
        print(f"\nFresh val file not found: {args.fresh_val_file} -- stopping before final report.")
        return

    fresh_entries = load_lexicon(args.fresh_val_file)
    print(f"\nLoaded {len(fresh_entries)} FRESH VAL words from {args.fresh_val_file}")
    fresh_rows = run_both_systems(fresh_entries, model, vocabs, device)

    f_rule_acc, f_neural_acc, f_oracle_acc = report_summary(fresh_rows, "FRESH VAL (final report, threshold frozen)")
    f_hybrid_acc = policy_accuracy(fresh_rows, best_t)
    print(f"\n  FROZEN threshold from VAL: {best_t:.2f}")
    print(f"  Hybrid accuracy on FRESH VAL: {f_hybrid_acc:.4f}")
    print(f"  Neural-only accuracy on FRESH VAL: {f_neural_acc:.4f}")
    print(f"  Gain: {f_hybrid_acc - f_neural_acc:+.4f}")
    print(f"  (This is the number to report. It was not used to pick anything.)")

    with open(args.dump_fresh_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["word", "gold", "rule_pred", "neural_pred", "confidence", "winner"])
        for r in fresh_rows:
            w.writerow([r["word"], " ".join(r["gold"]), " ".join(r["rule_pred"]),
                        " ".join(r["neural_pred"]), f"{r['confidence']:.4f}", r["winner"]])
    print(f"Dumped FRESH VAL rows to {args.dump_fresh_csv}")


if __name__ == "__main__":
    main()
