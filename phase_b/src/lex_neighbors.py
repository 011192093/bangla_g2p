import bisect

NUKTA_REMAP = {'ড়': 'ডz', 'ঢ়': 'ঢz', 'য়': 'যz'}

def sortkey(word):
    return ''.join(NUKTA_REMAP.get(ch, ch) for ch in word)

def load_sorted_lexicon(path="../data/lexicon.tsv"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            rows.append(tuple(parts))
    rows.sort(key=lambda r: sortkey(r[0]))
    keys = [sortkey(r[0]) for r in rows]
    return rows, keys

def neighbors(word, rows, keys, n=5):
    k = sortkey(word)
    i = bisect.bisect_left(keys, k)
    lo = max(0, i - n)
    hi = min(len(rows), i + n)
    return rows[lo:hi]

if __name__ == "__main__":
    import sys
    rows, keys = load_sorted_lexicon()
    for w in sys.argv[1:]:
        print("---", w, "---")
        for r in neighbors(w, rows, keys):
            print(r)
