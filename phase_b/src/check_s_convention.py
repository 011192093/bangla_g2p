import collections

src = "../data/bn_lexicon_normalized.txt"
pairs = []
for line in open(src, encoding="utf-8-sig"):
    p = line.split()
    if len(p) >= 2:
        pairs.append((p[0], p[1]))

# 1. Common native words that every speaker says with /ʃ/
for w in ["সকাল", "সময়", "সব", "সাথে", "সে", "সাত", "সবুজ", "সুন্দর", "সত্য", "সংসার"]:
    resp = [r for word, r in pairs if word == w]
    print(f"{w:<8} -> {resp[0] if resp else '(not in SUSTlex)'}")

# 2. Over all words starting with স: what does the respelling start with?
c = collections.Counter()
for word, r in pairs:
    if word.startswith("স") and not word.startswith(("স্", "সৃ")):
        c[r[0]] += 1
print("\nwords starting with স (not a cluster): respelling starts with")
for k, v in c.most_common(5):
    print(f"   {k}: {v}")
    