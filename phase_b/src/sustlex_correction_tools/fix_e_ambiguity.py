"""
Automated এ/ে (E/e) correction for SUSTlex-derived entries.

SUST's respelling doesn't distinguish open "ae" (E) from closed "e" --
it uses ে uniformly. This project already derived, tested, and
validated the actual rule (documented earlier this session):

  Within the WORD-INITIAL syllable, ে is "ae" (open) when:
    1. The next consonant carries া or ো, AND that consonant is the
       word's LAST syllable (দেখা "dEkha", তেরো "tEro")
    2. The next consonant carries no explicit vowel sign (inherent o)
       and is a bare SONORANT, AND that sonorant is not itself
       word-final (কেমন "kEmon", but তেল "tel" stays closed)
  Every non-initial syllable's ে is always closed "e".
  Everything else is plain "e".

This script applies that rule directly to the ORIGINAL WORD SPELLING
(not the SUST respelling) to determine E vs e, then patches the
SUSTlex-derived phoneme sequence at the corresponding position --
automating what would otherwise be 12,000+ manual checks.

NOTE: this reimplements the rule from this session's documentation.
If your actual g2p_engine.py has since been refined further, prefer
running YOUR live rule engine on these words instead of this
standalone reimplementation -- this script is a fallback for when
that's not convenient to wire up.
"""
import sys

SONORANTS = {"ঙ", "ঞ", "ণ", "ন", "ম", "ল", "র"}
HASANTA = "্"
VOWEL_SIGNS_WITH_INHERENT_O_BLOCKERS = {"া", "ি", "ী", "ু", "ূ", "ে", "ৈ", "ো", "ৌ"}


def get_clusters(word):
    """Simple cluster segmentation: returns list of
    (cluster_str, start_idx, end_idx, vowel_sign_or_None)."""
    clusters = []
    i, n = 0, len(word)
    CONS = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়"
    while i < n:
        ch = word[i]
        if ch in CONS:
            start = i
            chars = [ch]
            i += 1
            while i + 1 < n and word[i] == HASANTA and word[i+1] in CONS:
                chars.append(word[i+1])
                i += 2
            vowel_sign = None
            if i < n and word[i] in "ািীুূেৈোৌ":
                vowel_sign = word[i]
                i += 1
            clusters.append(("".join(chars), start, i, vowel_sign))
        else:
            i += 1
    return clusters


def should_be_open_E(word):
    """
    Returns True if this word's FIRST syllable ে should be the open
    'ae' reading, per this project's validated rule. Only meaningful
    for words whose first syllable actually has a bare consonant + ে.
    """
    clusters = get_clusters(word)
    if not clusters:
        return False

    first_cluster, start, end, vowel_sign = clusters[0]
    if vowel_sign != "ে":
        return False  # not applicable, first syllable isn't ে at all

    if len(clusters) < 2:
        return False  # no next syllable to check condition against

    next_cluster, next_start, next_end, next_vowel = clusters[1]
    is_word_final_syllable = (len(clusters) == 2)

    # Condition 1: next consonant carries া or ো AND is word's last syllable
    if next_vowel in ("া", "ো") and is_word_final_syllable:
        return True

    # Condition 2: next consonant is bare sonorant, not itself word-final
    n_consonants_in_next = len(next_cluster.split(HASANTA)) if HASANTA in next_cluster else 1
    is_bare = next_vowel is None
    is_sonorant = any(c in SONORANTS for c in next_cluster)
    if is_bare and is_sonorant and not is_word_final_syllable:
        return True

    return False


def patch_first_e_token(phoneme_list, use_E):
    """Given a phoneme list where the first 'e' token represents the
    first syllable's vowel, patch it to 'E' if needed."""
    patched = phoneme_list[:]
    for idx, tok in enumerate(patched):
        if tok == "e":
            if use_E:
                patched[idx] = "E"
            break  # only patch the FIRST e (first syllable), per the rule's scope
        elif tok in ("o", "O", "a", "i", "u"):
            break  # first syllable's vowel is something else, nothing to patch
    return patched


def process_file(input_path, output_path):
    corrected = 0
    unchanged = 0
    with open(input_path, encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.rstrip('\n')
            if line.startswith('#') or not line.strip():
                fout.write(line + '\n')
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                fout.write(line + '\n')
                continue
            word, phon_str = parts[0], parts[1]
            rest = parts[2:] if len(parts) > 2 else [""]

            use_E = should_be_open_E(word)
            variants = phon_str.split(" | ")
            new_variants = []
            for v in variants:
                tokens = v.split()
                new_tokens = patch_first_e_token(tokens, use_E)
                new_variants.append(" ".join(new_tokens))
                if new_tokens != tokens:
                    corrected += 1
                else:
                    unchanged += 1

            fout.write("\t".join([word, " | ".join(new_variants)] + rest) + "\n")

    print(f"Corrected (e -> E applied): {corrected}")
    print(f"Unchanged: {unchanged}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_e_ambiguity.py <input.tsv> <output.tsv>")
        sys.exit(1)
    process_file(sys.argv[1], sys.argv[2])
