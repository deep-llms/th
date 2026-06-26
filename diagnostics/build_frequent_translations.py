"""Build translation pairs from frequent training words + MUSE dictionaries.

Two directions:
1. xx→en: frequent non-English words → English translations
2. en→xx: frequent English words → non-English translations

Usage:
  python diagnostics/build_frequent_translations.py
  python diagnostics/build_frequent_translations.py --min-count 50
"""

import argparse
import json
import os
import sys
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MUSE_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries"
TARGET_LANGS = ["vi", "zh", "ru", "de", "ar"]


def download_muse_dict(src, tgt, cache_dir="temp/muse_dicts"):
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{src}-{tgt}.txt"
    path = os.path.join(cache_dir, filename)
    if not os.path.isfile(path):
        url = f"{MUSE_URL}/{filename}"
        print(f"  Downloading {url}...")
        urllib.request.urlretrieve(url, path)
    return path


def load_muse_dict(path):
    """Load MUSE dictionary. Returns dict: source_word -> list of target translations."""
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src = parts[0]
                tgt = parts[1]
                if src not in d:
                    d[src] = []
                d[src].append(tgt)
    return d


def load_freq_data(freq_file):
    with open(freq_file) as f:
        return json.load(f)


def get_frequent_words(lang_counts, min_count):
    return {w: c for w, c in lang_counts.items() if c >= min_count}


def get_word_counts(freq_data, lang):
    """Get word counts for a language. Falls back to 'all' if per-language not available."""
    per_lang = freq_data["per_lang"]
    if lang in per_lang:
        return per_lang[lang]["word_counts"]
    if "all" in per_lang:
        return per_lang["all"]["word_counts"]
    return None


def build_xx_to_en(freq_data, min_count):
    """Direction 1: frequent non-English words → English translations via MUSE xx→en."""
    all_pairs = {}

    for lang in TARGET_LANGS:
        lang_counts = get_word_counts(freq_data, lang)
        if lang_counts is None:
            print(f"[{lang}→en] No frequency data, skipping.")
            continue
        frequent_words = get_frequent_words(lang_counts, min_count)
        print(f"\n[{lang}→en] {len(frequent_words)} frequent words (>= {min_count} occurrences)")

        dict_path = download_muse_dict(lang, "en")
        muse = load_muse_dict(dict_path)
        print(f"[{lang}→en] MUSE entries: {len(muse)}")

        pairs = []
        for word, count in sorted(frequent_words.items(), key=lambda x: -x[1]):
            if word in muse:
                pairs.append({
                    lang: word,
                    f"{lang}_count": count,
                    "en": muse[word][0],
                    "en_all": muse[word],
                })

        all_pairs[lang] = pairs
        print(f"[{lang}→en] {len(pairs)} pairs with MUSE translations")
        if pairs:
            for p in pairs[:5]:
                print(f"  {p[lang]:<15} → {p['en']:<15} (count: {p[f'{lang}_count']:>8,})")
            if len(pairs) > 5:
                print(f"  ... and {len(pairs) - 5} more")

    return all_pairs


def build_en_to_xx(freq_data, min_count):
    """Direction 2: frequent English words → non-English translations via MUSE en→xx."""
    en_counts = get_word_counts(freq_data, "en")
    if en_counts is None:
        print("[en→xx] No English frequency data, skipping.")
        return {}
    frequent_en = get_frequent_words(en_counts, min_count)
    print(f"\n[en→xx] {len(frequent_en)} frequent English words (>= {min_count} occurrences)")

    all_pairs = {}

    for lang in TARGET_LANGS:
        dict_path = download_muse_dict("en", lang)
        muse = load_muse_dict(dict_path)
        print(f"[en→{lang}] MUSE entries: {len(muse)}")

        pairs = []
        for word, count in sorted(frequent_en.items(), key=lambda x: -x[1]):
            if word in muse:
                pairs.append({
                    "en": word,
                    "en_count": count,
                    lang: muse[word][0],
                    f"{lang}_all": muse[word],
                })

        all_pairs[lang] = pairs
        print(f"[en→{lang}] {len(pairs)} pairs")
        if pairs:
            for p in pairs[:5]:
                print(f"  {p['en']:<15} → {p[lang]:<15} (en count: {p['en_count']:>8,})")
            if len(pairs) > 5:
                print(f"  ... and {len(pairs) - 5} more")

    return all_pairs


def build_cross_lang_tuples(xx_en_pairs, en_xx_pairs):
    """Find words with translations across multiple languages."""
    # From xx→en: group by English translation
    en_to_langs_xxen = {}
    for lang, pairs in xx_en_pairs.items():
        for p in pairs:
            en = p["en"].lower()
            if en not in en_to_langs_xxen:
                en_to_langs_xxen[en] = {}
            en_to_langs_xxen[en][lang] = p[lang]

    # From en→xx: group by English source word
    en_to_langs_enxx = {}
    for lang, pairs in en_xx_pairs.items():
        for p in pairs:
            en = p["en"].lower()
            if en not in en_to_langs_enxx:
                en_to_langs_enxx[en] = {}
            en_to_langs_enxx[en][lang] = p[lang]

    # Merge both directions
    en_to_langs_merged = {}
    all_en_words = set(en_to_langs_xxen.keys()) | set(en_to_langs_enxx.keys())
    for en in all_en_words:
        merged = {}
        if en in en_to_langs_xxen:
            merged.update(en_to_langs_xxen[en])
        if en in en_to_langs_enxx:
            for lang, word in en_to_langs_enxx[en].items():
                if lang not in merged:
                    merged[lang] = word
        en_to_langs_merged[en] = merged

    multi_lang = {en: langs for en, langs in en_to_langs_merged.items() if len(langs) >= 2}
    all_five = {en: langs for en, langs in en_to_langs_merged.items() if len(langs) == 5}

    return en_to_langs_merged, multi_lang, all_five


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq-file", default="temp/frequent_words.json",
                        help="Output from find_frequent_words.py")
    parser.add_argument("--min-count", type=int, default=50,
                        help="Min occurrences to consider frequent (default: 50)")
    parser.add_argument("--output", default="temp/frequent_translations.json")
    args = parser.parse_args()

    freq_data = load_freq_data(args.freq_file)

    # Direction 1: xx→en
    print("=" * 60)
    print("DIRECTION 1: xx → en (frequent non-English → English)")
    print("=" * 60)
    xx_en_pairs = build_xx_to_en(freq_data, args.min_count)

    # Direction 2: en→xx
    print("\n" + "=" * 60)
    print("DIRECTION 2: en → xx (frequent English → non-English)")
    print("=" * 60)
    en_xx_pairs = build_en_to_xx(freq_data, args.min_count)

    # Cross-language tuples
    print("\n" + "=" * 60)
    print("CROSS-LANGUAGE TUPLES (merged from both directions)")
    print("=" * 60)

    en_to_langs, multi_lang, all_five = build_cross_lang_tuples(xx_en_pairs, en_xx_pairs)

    print(f"\n  Total English words with any translation: {len(en_to_langs)}")
    print(f"  English words with translations in >= 2 languages: {len(multi_lang)}")
    print(f"  English words with translations in ALL 5 languages: {len(all_five)}")

    for en, langs in sorted(all_five.items())[:15]:
        langs_str = " | ".join(f"{l}:{langs[l]}" for l in TARGET_LANGS if l in langs)
        print(f"    {en:<15} → {langs_str}")
    if len(all_five) > 15:
        print(f"    ... and {len(all_five) - 15} more")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("  xx→en pairs:")
    for lang, pairs in xx_en_pairs.items():
        print(f"    {lang}→en: {len(pairs)} pairs")
    print("  en→xx pairs:")
    for lang, pairs in en_xx_pairs.items():
        print(f"    en→{lang}: {len(pairs)} pairs")
    print(f"  Multi-lang (>=2): {len(multi_lang)}")
    print(f"  All-5-lang: {len(all_five)}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "min_count": args.min_count,
        "xx_en_pairs": {lang: pairs for lang, pairs in xx_en_pairs.items()},
        "xx_en_counts": {lang: len(pairs) for lang, pairs in xx_en_pairs.items()},
        "en_xx_pairs": {lang: pairs for lang, pairs in en_xx_pairs.items()},
        "en_xx_counts": {lang: len(pairs) for lang, pairs in en_xx_pairs.items()},
        "multi_lang_tuples": multi_lang,
        "all_five_tuples": all_five,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
