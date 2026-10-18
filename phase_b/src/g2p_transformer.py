#!/usr/bin/env python
"""
Transformer encoder-decoder G2P for Bangla (baseline to compare with the BiGRU).

Standalone: does not import your model.py / train.py, so it cannot break them.

Data format (same as your splits): word<TAB>phonemes[<TAB>register]
  - phonemes are split on spaces if the train file has spaces in them, else per character
    (override with --phon_sep).
  - register column is optional; missing -> "none".

Project rules built in:
  - Splits are read, never written or reshuffled (only train is shuffled per epoch).
  - Early stopping and checkpoint selection use VAL only.
  - Test is evaluated ONLY with --eval_test (and is never used during training).
  - Input normalization: norm_bn (composed nukta + NFC) always; Ahmad-style letter
    merging (ী→ি, ঈ→ই, ূ→ু, ঊ→উ, ষ→শ, ণ→ন) only with --ahmad_norm, input side only.

Usage (from ...\\phase_b\\src):
  python g2p_transformer.py train --train ../data/lexicon_split_train.tsv \
      --val ../data/lexicon_split_val.tsv --out_dir ../checkpoints_transformer --seed 1

  python g2p_transformer.py eval --checkpoint ../checkpoints_transformer/best_model.pt \
      --eval_file ../data/lexicon_fresh_val.tsv --dump fresh_val_transformer.tsv

  python g2p_transformer.py eval --checkpoint ... --eval_file ../data/lexicon_split_test.tsv \
      --eval_test --dump test_transformer.tsv          # final run only

  python g2p_transformer.py predict --checkpoint ... --words প্রস্ফুটিত বিস্ফোরণ
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import unicodedata

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="enable_nested_tensor")

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

# ---------------------------------------------------------------- normalization
NUKTA_FIX = {
    "য়": "য়",  # য + ় -> য়
    "ড়": "ড়",  # ড + ় -> ড়
    "ঢ়": "ঢ়",  # ঢ + ় -> ঢ়
}
AHMAD_MERGE = {
    "ী": "ি",  # ী -> ি
    "ঈ": "ই",  # ঈ -> ই
    "ূ": "ু",  # ূ -> ু
    "ঊ": "উ",  # ঊ -> উ
    "ষ": "শ",  # ষ -> শ
    "ণ": "ন",  # ণ -> ন
}


def norm_bn(w, ahmad=False):
    w = unicodedata.normalize("NFC", w.strip())
    for a, b in NUKTA_FIX.items():
        w = w.replace(a, b)
    if ahmad:
        for a, b in AHMAD_MERGE.items():
            w = w.replace(a, b)
    return w


# ---------------------------------------------------------------- data
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            if i == 0 and parts[0].strip().lower() in ("word", "grapheme", "graphemes"):
                continue  # header
            reg = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "none"
            rows.append((parts[0].strip(), parts[1].strip(), reg))
    return rows


def split_phon(p, sep):
    return p.split() if sep == "space" else list(p.replace(" ", ""))


class Vocab:
    def __init__(self, tokens):
        self.itos = SPECIALS + sorted(set(tokens) - set(SPECIALS))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, toks):
        u = self.stoi[UNK]
        return [self.stoi.get(t, u) for t in toks]

    def __len__(self):
        return len(self.itos)


def make_examples(rows, cfg, src_v, tgt_v, reg_v):
    ex = []
    for w, p, r in rows:
        s = src_v.encode(list(norm_bn(w, cfg["ahmad_norm"])))
        t = tgt_v.encode(split_phon(p, cfg["phon_sep"]))
        ex.append((s, t, reg_v.get(r, reg_v["<unk>"])))
    return ex


def batches(examples, bs, shuffle, rng=None):
    idx = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idx)
        # bucket by length for speed, keep randomness between buckets
        chunks = [idx[i:i + bs * 50] for i in range(0, len(idx), bs * 50)]
        idx = [j for c in chunks for j in sorted(c, key=lambda k: len(examples[k][0]))]
        groups = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        rng.shuffle(groups)
    else:
        groups = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    for g in groups:
        yield g


def collate(examples, ids, device):
    pad = 0
    S = max(len(examples[i][0]) for i in ids)
    T = max(len(examples[i][1]) for i in ids) + 1
    src = torch.full((len(ids), S), pad, dtype=torch.long)
    tin = torch.full((len(ids), T), pad, dtype=torch.long)
    tout = torch.full((len(ids), T), pad, dtype=torch.long)
    reg = torch.zeros(len(ids), dtype=torch.long)
    for b, i in enumerate(ids):
        s, t, r = examples[i]
        src[b, :len(s)] = torch.tensor(s)
        tin[b, :len(t) + 1] = torch.tensor([1] + t)        # BOS + t
        tout[b, :len(t) + 1] = torch.tensor(t + [2])       # t + EOS
        reg[b] = r
    return src.to(device), tin.to(device), tout.to(device), reg.to(device)


# ---------------------------------------------------------------- model
class PosEnc(nn.Module):
    def __init__(self, d, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TransformerG2P(nn.Module):
    def __init__(self, n_src, n_tgt, n_reg, d_model=256, nhead=4, enc_layers=3,
                 dec_layers=3, ff=1024, dropout=0.3):
        super().__init__()
        self.d = d_model
        self.src_emb = nn.Embedding(n_src, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(n_tgt, d_model, padding_idx=0)
        self.reg_emb = nn.Embedding(n_reg, d_model)
        self.pos = PosEnc(d_model)
        self.drop = nn.Dropout(dropout)
        self.tf = nn.Transformer(d_model, nhead, enc_layers, dec_layers, ff, dropout,
                                 batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, n_tgt)
        # Scale embeddings so emb*sqrt(d) has unit variance; otherwise the sinusoidal
        # positions are drowned out and the decoder loses track of where it is.
        for emb in (self.src_emb, self.tgt_emb, self.reg_emb):
            nn.init.normal_(emb.weight, 0.0, d_model ** -0.5)
        with torch.no_grad():
            self.src_emb.weight[0].zero_()
            self.tgt_emb.weight[0].zero_()

    def encode(self, src, reg):
        x = (self.src_emb(src) + self.reg_emb(reg).unsqueeze(1)) * math.sqrt(self.d)
        x = self.drop(self.pos(x))
        mask = src.eq(0)
        return self.tf.encoder(x, src_key_padding_mask=mask), mask

    def decode(self, tin, mem, src_mask):
        y = self.drop(self.pos(self.tgt_emb(tin) * math.sqrt(self.d)))
        causal = torch.triu(torch.ones(tin.size(1), tin.size(1), dtype=torch.bool,
                                       device=tin.device), 1)
        h = self.tf.decoder(y, mem, tgt_mask=causal, tgt_key_padding_mask=tin.eq(0),
                            memory_key_padding_mask=src_mask)
        return self.out(h)

    def forward(self, src, tin, reg):
        mem, m = self.encode(src, reg)
        return self.decode(tin, mem, m)

    @torch.no_grad()
    def greedy(self, src, reg, max_len=64):
        """Returns token lists and confidence = geometric mean of chosen-token probs
        (incl. EOS), same definition as your GRU confidence."""
        mem, m = self.encode(src, reg)
        B = src.size(0)
        ys = torch.full((B, 1), 1, dtype=torch.long, device=src.device)
        logp_sum = torch.zeros(B, device=src.device)
        n = torch.zeros(B, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(ys, mem, m)[:, -1]
            logits[:, [0, 1, 3]] = float("-inf")  # never emit <pad>, <s>, <unk>
            lp = F.log_softmax(logits, -1)
            lp_tok, tok = lp.max(-1)
            tok = tok.masked_fill(done, 0)
            logp_sum += lp_tok.masked_fill(done, 0.0)
            n += (~done).float()
            ys = torch.cat([ys, tok.unsqueeze(1)], 1)
            done |= tok.eq(2)
            if done.all():
                break
        out = []
        for b in range(B):
            seq = []
            for t in ys[b, 1:].tolist():
                if t in (0, 2):
                    break
                seq.append(t)
            out.append(seq)
        conf = torch.exp(logp_sum / n.clamp(min=1)).tolist()
        return out, conf


# ---------------------------------------------------------------- eval helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def run_eval(model, rows, examples, tgt_v, cfg, device, bs=512):
    model.eval()
    preds, confs = [], []
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    res = {}
    for i in range(0, len(order), bs):
        ids = order[i:i + bs]
        src, _, _, reg = collate(examples, ids, device)
        seqs, cf = model.greedy(src, reg)
        for j, s, c in zip(ids, seqs, cf):
            res[j] = (s, c)
    joiner = " " if cfg["phon_sep"] == "space" else ""
    correct = 0
    out = []
    for j, (w, gold, r) in enumerate(rows):
        s, c = res[j]
        pred = joiner.join(tgt_v.itos[t] for t in s)
        g = joiner.join(split_phon(gold, cfg["phon_sep"]))
        ok = pred == g
        correct += ok
        out.append((w, g, pred, c, int(ok), r))
    return correct, out


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    src_v, tgt_v = Vocab([]), Vocab([])
    src_v.itos, tgt_v.itos = ck["src_itos"], ck["tgt_itos"]
    src_v.stoi = {t: i for i, t in enumerate(src_v.itos)}
    tgt_v.stoi = {t: i for i, t in enumerate(tgt_v.itos)}
    reg_v = ck["reg_vocab"]
    model = TransformerG2P(len(src_v), len(tgt_v), len(reg_v), **cfg["model"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, src_v, tgt_v, reg_v, ck


# ---------------------------------------------------------------- commands
def cmd_train(a):
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    best_path = os.path.join(a.out_dir, "best_model.pt")
    if os.path.exists(best_path) and not a.force:
        sys.exit(f"{best_path} exists. Use a new --out_dir or pass --force.")

    tr_rows, va_rows = read_tsv(a.train), read_tsv(a.val)
    if a.phon_sep == "auto":
        sep = "space" if sum(" " in p for _, p, _ in tr_rows) > len(tr_rows) * 0.5 else "char"
    else:
        sep = a.phon_sep
    cfg = {
        "ahmad_norm": a.ahmad_norm, "phon_sep": sep,
        "model": dict(d_model=a.d_model, nhead=a.nhead, enc_layers=a.layers,
                      dec_layers=a.layers, ff=a.ff, dropout=a.dropout),
    }
    src_v = Vocab(c for w, _, _ in tr_rows for c in norm_bn(w, a.ahmad_norm))
    tgt_v = Vocab(t for _, p, _ in tr_rows for t in split_phon(p, sep))
    regs = sorted({r for _, _, r in tr_rows})
    reg_v = {r: i for i, r in enumerate(["<unk>"] + regs)}
    tr = make_examples(tr_rows, cfg, src_v, tgt_v, reg_v)
    va = make_examples(va_rows, cfg, src_v, tgt_v, reg_v)

    model = TransformerG2P(len(src_v), len(tgt_v), len(reg_v), **cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  phon_sep={sep}  ahmad_norm={a.ahmad_norm}")
    print(f"train={len(tr)} val={len(va)}  src_vocab={len(src_v)} tgt_vocab={len(tgt_v)} "
          f"registers={regs}  params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tr) / a.batch_size)
    total = steps_per_epoch * a.max_epochs
    warm = steps_per_epoch * a.warmup_epochs

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    meta = {
        "train_sha256": sha256(a.train), "val_sha256": sha256(a.val),
        "args": {k: v for k, v in vars(a).items() if k != "func"}, "n_params": n_params,
    }
    with open(os.path.join(a.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log = open(os.path.join(a.out_dir, "train_log.csv"), "w", newline="", encoding="utf-8")
    lw = csv.writer(log)
    lw.writerow(["epoch", "train_loss", "val_acc", "lr", "secs"])

    rng = random.Random(a.seed)
    best, best_ep, bad = -1.0, 0, 0
    for ep in range(1, a.max_epochs + 1):
        model.train()
        t0, tot, cnt = time.time(), 0.0, 0
        for ids in batches(tr, a.batch_size, True, rng):
            src, tin, tout, reg = collate(tr, ids, device)
            logits = model(src, tin, reg)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tout.reshape(-1),
                                   ignore_index=0, label_smoothing=a.label_smoothing)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(ids)
            cnt += len(ids)
        k, _ = run_eval(model, va_rows, va, tgt_v, cfg, device)
        acc = k / len(va)
        secs = time.time() - t0
        lw.writerow([ep, f"{tot / cnt:.4f}", f"{acc:.4f}", f"{sched.get_last_lr()[0]:.2e}",
                     f"{secs:.0f}"])
        log.flush()
        flag = ""
        if acc > best:
            best, best_ep, bad = acc, ep, 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                        "src_itos": src_v.itos, "tgt_itos": tgt_v.itos, "reg_vocab": reg_v,
                        "epoch": ep, "val_acc": acc, "meta": meta}, best_path)
            flag = "  *saved"
        else:
            bad += 1
        print(f"ep {ep:3d}  loss {tot / cnt:.4f}  val {acc:.4f} ({k}/{len(va)})  "
              f"{secs:.0f}s{flag}", flush=True)
        if bad >= a.patience and ep >= a.min_epochs:
            print(f"early stop: no val gain for {a.patience} epochs")
            break
    log.close()
    lo, hi = wilson(round(best * len(va)), len(va))
    print(f"best val {best:.4f} at epoch {best_ep}  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"checkpoint: {best_path}  sha256={sha256(best_path)}")
    print("Test was NOT evaluated. Use `eval --eval_test` once, at the very end.")


def cmd_eval(a):
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if "test" in os.path.basename(a.eval_file).lower() and not a.eval_test:
        sys.exit("This looks like the test split. Test is used once, for the final number. "
                 "Re-run with --eval_test if this is that run.")
    model, cfg, src_v, tgt_v, reg_v, ck = load_ckpt(a.checkpoint, device)
    rows = read_tsv(a.eval_file)
    ex = make_examples(rows, cfg, src_v, tgt_v, reg_v)
    k, out = run_eval(model, rows, ex, tgt_v, cfg, device)
    n = len(rows)
    lo, hi = wilson(k, n)
    print(f"{a.eval_file}: {k}/{n} = {k / n:.4f}  (95% Wilson CI {lo:.4f}-{hi:.4f})")
    print(f"file sha256={sha256(a.eval_file)}  checkpoint epoch={ck.get('epoch')}")
    unk = sum(1 for s, _, _ in ex if src_v.stoi[UNK] in s)
    if unk:
        print(f"note: {unk} words contain characters unseen in train (mapped to <unk>)")
    if a.dump:
        with open(a.dump, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["word", "gold", "pred", "confidence", "correct", "register"])
            for r in out:
                w.writerow([r[0], r[1], r[2], f"{r[3]:.4f}", r[4], r[5]])
        print(f"dump written: {a.dump}")


def cmd_predict(a):
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    model, cfg, src_v, tgt_v, reg_v, _ = load_ckpt(a.checkpoint, device)
    rows = [(w, "", a.register) for w in a.words]
    ex = make_examples(rows, cfg, src_v, tgt_v, reg_v)
    src, _, _, reg = collate(ex, list(range(len(ex))), device)
    seqs, cf = model.greedy(src, reg)
    joiner = " " if cfg["phon_sep"] == "space" else ""
    for w, s, c in zip(a.words, seqs, cf):
        print(f"{w}\t{joiner.join(tgt_v.itos[t] for t in s)}\tconf={c:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--train", required=True)
    t.add_argument("--val", required=True)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--d_model", type=int, default=256)
    t.add_argument("--nhead", type=int, default=4)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--ff", type=int, default=1024)
    t.add_argument("--dropout", type=float, default=0.3)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--batch_size", type=int, default=256)
    t.add_argument("--max_epochs", type=int, default=80)
    t.add_argument("--warmup_epochs", type=int, default=2)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--min_epochs", type=int, default=10,
                   help="never early-stop before this epoch")
    t.add_argument("--label_smoothing", type=float, default=0.0,
                   help="0.1 often helps accuracy but lowers confidence scores")
    t.add_argument("--phon_sep", choices=["auto", "space", "char"], default="auto")
    t.add_argument("--ahmad_norm", action="store_true",
                   help="merge same-sound letters on the INPUT side (ablation)")
    t.add_argument("--force", action="store_true")
    t.add_argument("--cpu", action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--eval_file", required=True)
    e.add_argument("--dump")
    e.add_argument("--eval_test", action="store_true",
                   help="required to evaluate the test split (final run only)")
    e.add_argument("--cpu", action="store_true")
    e.set_defaults(func=cmd_eval)

    q = sub.add_parser("predict")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--words", nargs="+", required=True)
    q.add_argument("--register", default="none")
    q.add_argument("--cpu", action="store_true")
    q.set_defaults(func=cmd_predict)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
