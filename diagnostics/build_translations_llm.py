"""Build translation pairs using an LLM API (Gemini or OpenAI).

Replaces MUSE dictionaries with high-quality LLM translations that handle
multi-word expressions (e.g., Vietnamese "chính phủ" for "government").

Output format is compatible with anchor_probe2_muse_no_loan_word.py.

Usage:
  # Pass all step freq files — words are merged and deduped, translated once
  python diagnostics/build_translations_llm.py \
      --freq-files smoke_test_outputs/S3/token_ids/frequent_words_step*.json \
      --output temp/frequent_translations_llm.json \
      --provider openai

  # Resume interrupted run (reads existing output, skips already-translated words)
  python diagnostics/build_translations_llm.py \
      --freq-files ... --output temp/frequent_translations_llm.json
"""

import argparse
import json
import os
import sys
import time

TARGET_LANGS = ["vi", "zh", "ru", "de", "ar"]
LANG_NAMES = {"vi": "Vietnamese", "zh": "Chinese", "ru": "Russian", "de": "German", "ar": "Arabic"}

PROMPT_TEMPLATE = """Translate each English word into Vietnamese, Chinese (Simplified), Russian, German, and Arabic. Use the most natural translation a native speaker would use in everyday context. For words with multiple meanings, use the most common sense. If a word has no real translation (e.g. technical acronym), set its value to null.

Return ONLY a JSON object, no markdown, no explanation.

Words: {words}

Format: {{"word": {{"vi": "...", "zh": "...", "ru": "...", "de": "...", "ar": "..."}}, ...}}"""


def load_frequent_english(freq_files, min_count, max_words):
    """Load and merge English word counts across multiple freq files, dedup by word."""
    merged = {}
    for freq_file in freq_files:
        with open(freq_file) as f:
            data = json.load(f)
        per_lang = data.get("per_lang", {})
        if "en" not in per_lang:
            print(f"  WARNING: no English data in {freq_file}, skipping")
            continue
        for word, count in per_lang["en"]["word_counts"].items():
            key = word.lower()
            if key not in merged or count > merged[key][1]:
                merged[key] = (word, count)
        print(f"  Loaded {freq_file}: {len(per_lang['en']['word_counts']):,} English words")

    if not merged:
        print("ERROR: no English data found in any freq file")
        sys.exit(1)

    filtered = {word: count for _key, (word, count) in merged.items() if count >= min_count}
    sorted_words = sorted(filtered.items(), key=lambda x: -x[1])
    if max_words:
        sorted_words = sorted_words[:max_words]
    print(f"  After merge + dedup: {len(merged):,} unique words, {len(sorted_words)} after min_count={min_count} + max_words={max_words}")
    return sorted_words


def load_existing(output_path):
    if not os.path.isfile(output_path):
        return {}
    with open(output_path) as f:
        data = json.load(f)
    return data.get("all_five_tuples", {})


def call_gemini(words, model="gemini-2.0-flash", max_retries=3):
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY environment variable")
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)

    prompt = PROMPT_TEMPLATE.format(words=json.dumps(words))
    for attempt in range(max_retries):
        try:
            response = client.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def call_openai(words, model="gpt-4o-mini", max_retries=3):
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")
    client = OpenAI(api_key=api_key)

    prompt = PROMPT_TEMPLATE.format(words=json.dumps(words))
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def validate_entry(en_word, translations):
    """Check that all 5 target languages are present with real translations."""
    if not isinstance(translations, dict):
        return False
    for lang in TARGET_LANGS:
        val = translations.get(lang)
        if not val or not isinstance(val, str):
            return False
    return True


def save_output(all_five, output_path):
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    output = {
        "source": "llm",
        "all_five_tuples": all_five,
        "multi_lang_tuples": all_five,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq-files", nargs="+", required=True,
                        help="One or more freq files from count_words_from_text.py (words are merged and deduped)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--model", default=None,
                        help="Model name (default: gemini-2.0-flash or gpt-4o)")
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--max-words", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    if args.model is None:
        args.model = "gemini-2.0-flash" if args.provider == "gemini" else "gpt-4o"

    call_api = call_gemini if args.provider == "gemini" else call_openai

    sorted_words = load_frequent_english(args.freq_files, args.min_count, args.max_words)
    print(f"Frequent English words: {len(sorted_words)} (min_count={args.min_count}, max={args.max_words})")

    existing = load_existing(args.output)
    print(f"Already translated: {len(existing)} (will skip)")

    words_to_translate = [(w, c) for w, c in sorted_words if w.lower() not in existing]
    print(f"Remaining: {len(words_to_translate)}")

    if not words_to_translate:
        print("Nothing to do.")
        return

    all_five = dict(existing)
    total_batches = (len(words_to_translate) + args.batch_size - 1) // args.batch_size
    translated = 0
    failed = 0

    for batch_idx in range(0, len(words_to_translate), args.batch_size):
        batch = words_to_translate[batch_idx:batch_idx + args.batch_size]
        batch_words = [w for w, _ in batch]
        batch_num = batch_idx // args.batch_size + 1

        print(f"\n  Batch {batch_num}/{total_batches} ({len(batch_words)} words)...")
        result = call_api(batch_words, model=args.model)

        if result is None:
            print(f"    FAILED entire batch after retries")
            failed += len(batch_words)
            continue

        for en_word in batch_words:
            matches = [k for k in result if k.lower() == en_word.lower()]
            if matches and validate_entry(en_word, result[matches[0]]):
                translations = result[matches[0]]
                clean = {lang: translations[lang] for lang in TARGET_LANGS}
                all_five[en_word.lower()] = clean
                translated += 1
            else:
                failed += 1

        save_output(all_five, args.output)
        print(f"    Saved. Total: {len(all_five)} translated, {failed} failed")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Translated: {translated}")
    print(f"  Failed: {failed}")
    print(f"  Total tuples: {len(all_five)}")
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
