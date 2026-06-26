"""Count word frequencies in each language's training data.

Outputs full word counts so build_frequent_translations.py can match
against MUSE dictionaries (which have ~100K entries per language pair).

Usage:
  python diagnostics/find_frequent_words.py --data-dir data/Qwen_Qwen3-0.6B/train
  python diagnostics/find_frequent_words.py --data-dir data/Qwen_Qwen3-0.6B/train --max-docs 200000
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk

LANGS = ["en", "vi", "zh", "ru", "de", "ar"]


def count_words_whitespace(texts, report_lang="??", report_every=100_000):
    """Count words by whitespace splitting + punctuation stripping."""
    counts = Counter()
    for i, text in enumerate(texts):
        for token in text.split():
            cleaned = token.strip(".,;:!?\"'()[]{}—–-…·«»""''")
            if cleaned:
                counts[cleaned] += 1
        if (i + 1) % report_every == 0:
            print(f"  [{report_lang}] {i+1:,}/{len(texts):,} docs, {len(counts):,} unique words")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/Qwen_Qwen3-0.6B/train")
    parser.add_argument("--min-count", type=int, default=10,
                        help="Only save words with >= this many occurrences (default: 10)")
    parser.add_argument("--max-docs", type=int, default=100_000,
                        help="Max documents to scan per language (default: 100K)")
    parser.add_argument("--output", default="temp/frequent_words.json")
    args = parser.parse_args()

    results = {}

    for lang in LANGS:
        lang_dir = os.path.join(args.data_dir, lang)
        if not os.path.isdir(lang_dir):
            print(f"[{lang}] Not found at {lang_dir}, skipping.")
            continue

        print(f"\n[{lang}] Loading dataset...")
        ds = load_from_disk(lang_dir)
        total_docs = len(ds)
        n_docs = min(args.max_docs, total_docs)
        texts = ds["text"][:n_docs]
        print(f"[{lang}] Scanning {n_docs:,}/{total_docs:,} documents...")

        if lang == "zh":
            print(f"[{lang}] WARNING: whitespace splitting is unreliable for Chinese.")
            print(f"[{lang}] Chinese character-level words may be undercounted.")

        counts = count_words_whitespace(texts, report_lang=lang)

        # Filter to min_count
        filtered = {w: c for w, c in counts.items() if c >= args.min_count}

        results[lang] = {
            "total_docs": n_docs,
            "total_unique_words": len(counts),
            "words_above_threshold": len(filtered),
            "word_counts": filtered,
        }

        print(f"[{lang}] {len(counts):,} unique words total")
        print(f"[{lang}] {len(filtered):,} words with >= {args.min_count} occurrences")
        top5 = counts.most_common(5)
        print(f"[{lang}] Top 5: {top5}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "min_count_threshold": args.min_count,
            "max_docs_per_lang": args.max_docs,
            "per_lang": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
