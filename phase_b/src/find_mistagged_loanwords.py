"""
Heuristic (non-ML) detector for English-origin loanwords that are tagged
something other than loanword/loanword_nativized in lexicon.tsv.

Method: strip a trailing Bangla case/plural/classifier suffix (possibly
chained) off each candidate word, then check the remaining stem against a
hand-built reference list of common English loan-roots as they're commonly
transliterated into Bangla. This is necessarily incomplete (no ML, no
exhaustive dictionary) -- it's meant to surface a reviewable batch, not
guarantee full recall.

Also flags, for every match, whether its phonemes look consistent with the
project's established English-loanword s/sh convention: plain "s" before a
stop consonant, except the native-feeling "-ist/-ast/-ust" family which the
project keeps "sh" for independently (see earlier session fixes); "sh" here
on an otherwise-clean English loanword is flagged as a likely s/sh bug.

Usage:
    python find_mistagged_loanwords.py > candidates.tsv
"""
import io
import re

LEXICON = "../data/lexicon.tsv"

SKIP_TAGS = {"loanword", "loanword_nativized"}

# Suffixes stripped (longest first) before stem-matching. Chained once;
# good enough for the common single-suffix case this heuristic targets.
SUFFIXES = [
    "গুলোকে", "গুলোর", "গুলোও", "গুলোতে", "গুলো",
    "টিকে", "টির", "টিতে", "টিও", "টি",
    "টাকে", "টার", "টাতে", "টাও", "টা",
    "খানা", "খানি",
    "দেরকে", "দেরও", "দের",
    "কেও", "কেই", "কে",
    "এরও", "এর", "ের",
    "সহ", "হীন", "ভিত্তিক", "বিহীন", "ময়",
    "রা", "র", "তে", "ও", "ই", "এ",
]
SUFFIXES.sort(key=len, reverse=True)


def strip_suffix(word):
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf) + 1:
            return word[: -len(suf)]
    return word


# Hand-built reference list of common English loan-roots, spelled as they
# usually appear transliterated into Bangla. Not exhaustive -- a reviewable
# starting batch, grouped by domain for maintainability.
ENGLISH_ROOTS = set("""
প্রোটিন হাউস হাউসফুল ফুল্লহাউস কম্পিউটার ইন্টারনেট মোবাইল ফোন টেলিফোন
টেবিল স্কুল কলেজ বাস ট্রেন কার ব্যাংক বাংক হোটেল রেস্তোরা রেস্টুরেন্ট
ক্যামেরা টিভি ভিডিও রেডিও স্টেশন বাস্কেটবল ফুটবল ক্রিকেট হকি টেনিস
ভিটামিন ক্যালরি প্রোগ্রাম সফটওয়্যার হার্ডওয়্যার ইমেল ইমেইল ওয়েবসাইট
পাসওয়ার্ড লগইন আপলোড ডাউনলোড ইউজার একাউন্ট প্রোফাইল পোস্ট কমেন্ট
লাইক শেয়ার ফলোয়ার সাবস্ক্রাইব চ্যানেল ব্লগ নেটওয়ার্ক সার্ভার ডাটা
ডেটা ফাইল ফোল্ডার সিস্টেম প্রসেস প্রোগ্রামিং কোডিং ডিজাইন ইঞ্জিনিয়ার
ডাক্তার নার্স হাসপাতাল ক্লিনিক মেডিসিন অপারেশন সার্জারি ভ্যাকসিন
ইনজেকশন থার্মোমিটার স্টেথোস্কোপ এক্সরে স্কান টেস্ট রিপোর্ট
প্লেন এয়ারপোর্ট ফ্লাইট টিকিট পাসপোর্ট ভিসা হোটেল ট্যাক্সি বাইক
মোটরসাইকেল সাইকেল স্কুটার হেলমেট ব্রেক ইঞ্জিন টায়ার পেট্রোল ডিজেল
শার্ট প্যান্ট জ্যাকেট কোট সোয়েটার টিশার্ট জিন্স বেল্ট স্যান্ডেল
ব্যাগ ওয়ালেট পার্স গ্লাভস স্কার্ফ টাই সুট ড্রেস স্কার্ট
কেক বিস্কুট চকলেট আইসক্রিম স্যান্ডউইচ বার্গার পিজা পিৎজা স্যুপ সস
কফি চা টোস্ট জুস কোকাকোলা স্প্রাইট পপকর্ন চিপস নুডলস পাস্তা
ম্যানেজার ডিরেক্টর অফিসার সেক্রেটারি প্রেসিডেন্ট চেয়ারম্যান
সিইও কোম্পানি অফিস মিটিং প্রজেক্ট প্ল্যান বাজেট মার্কেট ইনভেস্টমেন্ট
ব্যাংকিং ফাইন্যান্স ইকোনমি ইন্ডাস্ট্রি ফ্যাক্টরি প্রডাক্ট ব্র্যান্ড
এডভার্টাইজমেন্ট মার্কেটিং সেলস কাস্টমার ক্লায়েন্ট
ফিল্ম সিনেমা মুভি থিয়েটার ডিরেক্টর প্রডিউসার
মিউজিক গিটার পিয়ানো ড্রাম ভায়োলিন কনসার্ট অ্যালবাম
টুর্নামেন্ট ম্যাচ পয়েন্ট স্কোর চ্যাম্পিয়ন টিম ক্যাপ্টেন কোচ
প্লেয়ার স্টেডিয়াম মেডাল ট্রফি অলিম্পিক
টাইম মিনিট সেকেন্ড ক্যালেন্ডার শিডিউল ডেডলাইন
ইউনিভার্সিটি কলেজ ক্যাম্পাস ক্লাস লেকচার প্রফেসর স্টুডেন্ট এক্সাম
সার্টিফিকেট ডিগ্রি ডিপ্লোমা স্কলারশিপ
পার্লামেন্ট মিনিস্টার প্রেসিডেন্ট পুলিশ আর্মি
পাসপোর্ট ভোট ইলেকশন ক্যাম্পেইন পলিসি
টেকনোলজি সায়েন্স ফিজিক্স কেমিস্ট্রি বায়োলজি ম্যাথ ম্যাথমেটিক্স
এনার্জি ইলেকট্রিসিটি সোলার পাওয়ার জেনারেটর ব্যাটারি চার্জার
রোবট স্যাটেলাইট রকেট মিসাইল ড্রোন
হোটেল রিসোর্ট ট্যুরিজম ট্যুর গাইড
চেয়ার গ্লাস পেন্সিল পেন মার্কার স্টেপলার ক্লিপ বোর্ড চক ব্রাশ শ্যাম্পু
লোশন ক্রিম পাউডার টিস্যু ন্যাপকিন কাপ প্লেট বোল স্পুন ফর্ক নাইফ
ফ্রিজ ওভেন মাইক্রোওভেন হিটার ফ্যান লাইট বাল্ব সুইচ ক্যাবল ওয়্যার
প্লাগ সকেট মাস্ক গগলস
টিচার ড্রাইভার পাইলট ক্যাশিয়ার ওয়েটার গার্ড সিকিউরিটি
শপিংমল মল পার্ক জিম ক্লাব বার ক্যাফে লাইব্রেরি
ল্যাপটপ কিবোর্ড মাউস স্ক্রিন মনিটর প্রিন্টার স্ক্যানার হেডফোন
স্পিকার মাইক্রোফোন মাইক চার্জ সিগনাল ওয়াইফাই ব্লুটুথ
আইডিয়া লজিক স্টাইল ফ্যাশন ট্রেন্ড পার্সোনালিটি ক্যারেক্টার হিরো
হিরোইন সুপারস্টার সেলিব্রিটি
রাগবি গলফ বক্সিং সুইমিং রানিং জগিং
""".split())

# fast normalize helper: words in the ref list may also show up with a
# stray trailing vowel-sign already baked in from how I typed them --
# exact-match only, deliberately conservative.


def check_s_sh(phon):
    """Flag 'sh' immediately before a stop consonant as a likely s/sh bug
    for an otherwise-English loanword, per this project's established
    convention (plain 's' in English loanwords, except the -ist/-ast
    family kept 'sh' separately -- too narrow a family to false-positive
    on here)."""
    toks = phon.split()
    flags = []
    for i in range(len(toks) - 1):
        if toks[i] == "sh" and toks[i + 1] in ("k", "T", "p", "t"):
            flags.append(f"sh+{toks[i+1]} at position {i}")
    return flags


def main():
    with io.open(LEXICON, encoding="utf-8") as f:
        lines = f.readlines()

    candidates = []
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        word, phon, tag = parts
        if tag in SKIP_TAGS:
            continue
        stem = strip_suffix(word)
        if stem in ENGLISH_ROOTS or word in ENGLISH_ROOTS:
            sh_flags = check_s_sh(phon)
            candidates.append((word, phon, tag, stem, sh_flags))

    print(f"{len(candidates)} candidates (current tag != loanword/loanword_nativized)\n")
    for word, phon, tag, stem, sh_flags in candidates:
        flag_str = f"  [s/sh? {'; '.join(sh_flags)}]" if sh_flags else ""
        print(f"{word}\t{phon}\t{tag}\t(stem matched: {stem}){flag_str}")


if __name__ == "__main__":
    main()
