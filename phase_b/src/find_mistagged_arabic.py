"""
Heuristic (non-ML) detector for Arabic/Persian-origin loanwords tagged
something other than loanword/loanword_nativized in lexicon.tsv.

Same method as find_mistagged_loanwords.py: strip a trailing Bangla
suffix, check the stem against a hand-built reference list of common
Arabic/Persian-origin words as transliterated into Bangla. Religious/
administrative vocabulary mostly stays "loanword"; words that are
specifically person names (not generic vocabulary) are reported
separately since they likely belong under proper_noun instead -- same
split as কোকাকোলা vs পিৎজা in the English pass.

Usage:
    python find_mistagged_arabic.py > candidates.tsv
"""
import io
from find_mistagged_loanwords import strip_suffix, SKIP_TAGS

LEXICON = "../data/lexicon.tsv"

# Generic Arabic/Persian-origin vocabulary (religious, administrative,
# everyday) -- candidates for tag "loanword".
ARABIC_VOCAB = set("""
মসজিদ ঈদ রোজা সেহরি ইফতার কুরআন কোরআন হাদিস ফতোয়া জিহাদ ইমাম মুফতি
ফিকাহ শরিয়া শরিয়ত হালাল হারাম জান্নাত জাহান্নাম দোয়া তাকবির এতিম
ওয়াক্ফ এবাদত ইবাদত নামাজ সালাত জাকাত হজ্জ হজ উমরাহ কাবা মক্কা মদিনা
তাওহিদ শিরক কুফর মুনাফিক মুসলিম মুসলমান কাফের ইসলাম দ্বীন মাদ্রাসা
মক্তব হুজুর মাওলানা মুহতারাম জনাব সাহেব খতিব মুয়াজ্জিন আজান তারাবি
জুমা জুমার কোরবানি আকিকা মিলাদ ওরস দরগা দরগাহ পীর মুরিদ তরিকা তাবিজ
ফরজ ওয়াজিব সুন্নত নফল মাকরুহ হারামজাদা তালাক নিকাহ মোহর ইদ্দত
ওয়ারিশ মিরাস খোদা আল্লাহ রাসুল নবি নবী পয়গম্বর ফেরেশতা
শয়তান ইবলিস জিন ফেতনা গজব রহমত বরকত কেরামত মুজিজা তাকদির নসিব
কিয়ামত হাশর মিজান পুলসিরাত বেহেশত দোজখ গুনাহ সওয়াব ফজিলত এবাদতখানা
মসনদ তখত সুলতান বাদশাহ নবাব আমির খলিফা উজির দিওয়ান কাজি
কোতোয়াল জমিদার তহসিলদার মুনসি উকিল ফরিয়াদি মুদ্দাই আসামি হাকিম
উজিরে এলাকা মহকুমা তালুক পরগনা জায়গির আমলা দফতর দরবার হুকুম ফরমান
তারিখ সনদ দলিল রসিদ হিসাব বাজেট মুনাফা নগদ বকেয়া কিস্তি
সুদ কারবার দোকান বাজার আড়ত মহাজনি দালাল কমিশন খরচ
আমানত জিম্মা হেফাজত হিফাজত আমানতদার জামানত জরিমানা কয়েদ
জেলখানা ফাঁসি কসাই কসাইখানা বাবুর্চি খানসামা দরজি মিস্ত্রি কারিগর
""".split())

# Words that are specifically names (person/place of Arabic/Persian
# origin), reported separately -- likely proper_noun, not loanword.
ARABIC_NAME_ELEMENTS = set("""
মুহাম্মদ মোহাম্মদ মুহাম্মাদ আহমদ আহমেদ আব্দুল আবদুল রহমান রহিম করিম
হাসান হোসেন হুসাইন ওমর উসমান আলি আলী ফাতিমা খাদিজা আয়েশা জয়নব
ইব্রাহিম ইউসুফ ইসমাইল ইসহাক দাউদ সুলাইমান মুসা ইয়াকুব ইদ্রিস
সালমান ইমরান জাকারিয়া ইয়াহিয়া ইলিয়াস ইউনুস আইয়ুব জুলকারনাইন
খালিদ তারিক জামাল কামাল জামিল নাসির ফারুক ইকবাল আশরাফ আকবর
নাজমুল শহিদুল রফিকুল হাবিবুর সিদ্দিক সিদ্দিকা হাফিজ হাফিজা
বেগম খাতুন সুলতানা জাহান পারভিন ফেরদৌসি নাসরিন ইয়াসমিন
বায়তুল বায়তুশ দারুল আইনুল নূরুল নুরুল শামসুল কামরুল
""".split())


def main():
    with io.open(LEXICON, encoding="utf-8") as f:
        lines = f.readlines()

    vocab_hits, name_hits = [], []
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        word, phon, tag = parts
        if tag in SKIP_TAGS:
            continue
        stem = strip_suffix(word)
        if stem in ARABIC_VOCAB or word in ARABIC_VOCAB:
            vocab_hits.append((word, phon, tag, stem))
        elif stem in ARABIC_NAME_ELEMENTS or word in ARABIC_NAME_ELEMENTS:
            name_hits.append((word, phon, tag, stem))

    print(f"{len(vocab_hits)} generic-vocabulary candidates (suggest -> loanword)\n")
    for word, phon, tag, stem in vocab_hits:
        print(f"{word}\t{phon}\t{tag}\t(stem: {stem})")

    print(f"\n{len(name_hits)} name-element candidates (suggest -> proper_noun)\n")
    for word, phon, tag, stem in name_hits:
        print(f"{word}\t{phon}\t{tag}\t(stem: {stem})")


if __name__ == "__main__":
    main()
