from canon_rescore import canon

FILES = {
    "test": "../data/frozen_2026-10-25/lexicon_split_test.tsv",
    "fresh-val": "../data/lexicon_fresh_val_v1_1.tsv",
}
for name, path in FILES.items():
    rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8-sig") if l.count("\t") >= 1]
    rows = [r for r in rows if r[0].strip().lower() != "word"]
    single = [len(canon(r[1]).split()) for r in rows if " " not in r[0].strip()]
    multi = [len(canon(r[1]).split()) for r in rows if " " in r[0].strip()]
    print(f"{name}: single words over 30 symbols: {sum(n > 30 for n in single)} of {len(single)}   "
          f"multi-word over 30: {sum(n > 30 for n in multi)} of {len(multi)}   "
          f"longest single word: {max(single)}")