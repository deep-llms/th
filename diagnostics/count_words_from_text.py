"""Step 2 of 2: detect language + count words from decoded text.

Reads text_rank{N}.jsonl (produced by decode_token_ids.py — one JSON-encoded
string per line) and meta.json. For each requested step it takes the first
S * seqs_per_step lines per rank, fans them out to a pool of fasttext workers
(each loads its own lid.176.bin) for language detection + word counting, and
accumulates results (step S = words seen up to S).

This script uses ONLY fasttext + multiprocessing — NO transformers — so there is
zero transformers-vs-multiprocessing interaction by construction.

Output: frequent_words_step{N}.json (compatible with build_frequent_translations.py).

Usage:
  python diagnostics/count_words_from_text.py \
      --input-dir /path/to/token_ids \
      --steps 1500 3250 5500 6500 --min-count 5 --workers 48
"""

import argparse
import json
import os
import urllib.request
import multiprocessing as mp
from collections import Counter

LANGS = ["en", "vi", "zh", "ru", "de", "ar"]
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_PATH = "temp/lid.176.bin"


def download_fasttext_model(path=FASTTEXT_MODEL_PATH):
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading fasttext lid model to {path}...")
    urllib.request.urlretrieve(FASTTEXT_MODEL_URL, path)
    print(f"Downloaded ({os.path.getsize(path) / 1e6:.0f}MB)")
    return path


def load_fasttext_model(path):
    try:
        import fasttext
    except ImportError:
        raise ImportError("fasttext not installed. Run: pip install fasttext-wheel")
    fasttext.FastText.eprint = lambda x: None
    return fasttext.load_model(path)


def count_words_from_text(text):
    """Split text into words and clean punctuation."""
    counts = Counter()
    for token in text.split():
        cleaned = token.strip(".,;:!?\"'()[]{}—–-…·«»""''")
        if cleaned:
            counts[cleaned] += 1
    return counts


def read_meta(input_dir):
    path = os.path.join(input_dir, "meta.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found (copied here by decode_token_ids.py).")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Worker: each process loads its OWN fasttext model (process-isolated).
# ---------------------------------------------------------------------------
_WORKER_FT = None


def _worker_init(model_path):
    global _WORKER_FT
    _WORKER_FT = load_fasttext_model(model_path)


def _worker_count(texts):
    """Detect language + count words for a list of texts. Plain single-string
    fasttext predict (well-tested API); parallelism comes from the POOL."""
    per_lang = {}
    all_counts = Counter()
    lang_seqs = Counter()
    for text in texts:
        clean = text.replace("\n", " ").strip()  # fasttext needs single-line input
        if not clean:
            lang = "other"
        else:
            label = _WORKER_FT.predict(clean, k=1)[0][0].replace("__label__", "")
            lang = label if label in LANGS else "other"
        wc = count_words_from_text(text)
        if wc:
            per_lang.setdefault(lang, Counter()).update(wc)
            all_counts.update(wc)
        lang_seqs[lang] += 1
    return lang_seqs, per_lang, all_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="temp/frequent_words_by_step",
                        help="Dir with text_rank*.jsonl + meta.json")
    parser.add_argument("--steps", nargs="+", type=int, default=[1500, 3250, 5500])
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--read-chunk", type=int, default=40000,
                        help="Sequences to gather (across ranks) before parallel dispatch")
    parser.add_argument("--fasttext-model", default=FASTTEXT_MODEL_PATH)
    args = parser.parse_args()

    meta = read_meta(args.input_dir)
    world_size = meta["world_size"]
    seqs_per_step = meta["seqs_per_step_per_rank"]

    model_path = download_fasttext_model(args.fasttext_model)
    print(f"Fasttext model: {model_path}")
    print(f"Meta: world_size={world_size}, seqs_per_step_per_rank={seqs_per_step}, workers={args.workers}")
    for step in sorted(args.steps):
        per_rank = step * seqs_per_step
        print(f"  Step {step}: {per_rank:,} seqs/rank x {world_size} ranks = {per_rank*world_size:,} sequences")

    # Open text files
    rank_files = {}
    rank_line_counts = {}
    for rank in range(world_size):
        path = os.path.join(args.input_dir, f"text_rank{rank}.jsonl")
        if not os.path.isfile(path):
            print(f"WARNING: missing {path}")
            continue
        rank_files[rank] = open(path, "r", encoding="utf-8")
        rank_line_counts[rank] = 0
    if not rank_files:
        print(f"ERROR: no text_rank*.jsonl in {args.input_dir} (run decode_token_ids.py first)")
        return

    # Worker pool (fasttext only — no transformers anywhere in this process tree,
    # so the default start method is safe; it is 'fork' on Linux).
    pool = mp.Pool(args.workers, initializer=_worker_init, initargs=(model_path,))

    per_lang_counts = {lang: Counter() for lang in LANGS}
    per_lang_counts["other"] = Counter()
    all_counts = Counter()
    lang_seq_counts = Counter()
    total = 0

    def process_chunk(texts):
        nonlocal total
        if not texts:
            return
        n = max(1, (len(texts) + args.workers - 1) // args.workers)
        sublists = [texts[i:i + n] for i in range(0, len(texts), n)]
        for lang_seqs_p, per_lang_p, all_c_p in pool.map(_worker_count, sublists):
            for lang, cnt in lang_seqs_p.items():
                lang_seq_counts[lang] += cnt
                total += cnt
            for lang, c in per_lang_p.items():
                per_lang_counts.setdefault(lang, Counter()).update(c)
            all_counts.update(all_c_p)

    sorted_steps = sorted(args.steps)
    prev_total = 0
    try:
        for step in sorted_steps:
            target_per_rank = step * seqs_per_step
            print(f"\nProcessing step {step} (up to {target_per_rank:,} seqs/rank)...")

            buffer = []
            for rank in sorted(rank_files.keys()):
                to_read = target_per_rank - rank_line_counts[rank]
                if to_read <= 0:
                    continue
                read = 0
                for _ in range(to_read):
                    line = rank_files[rank].readline()
                    if not line:
                        break
                    rank_line_counts[rank] += 1
                    read += 1
                    try:
                        text = json.loads(line)   # JSON-encoded string
                    except json.JSONDecodeError as e:
                        print(f"  WARNING: bad line rank {rank} line {rank_line_counts[rank]}: {e}")
                        text = ""
                    buffer.append(text)
                    if len(buffer) >= args.read_chunk:
                        process_chunk(buffer)
                        buffer = []
                        print(f"  {total:,} sequences processed...")
                if read < to_read:
                    print(f"  WARNING: rank {rank} ran out at {rank_line_counts[rank]:,} lines "
                          f"(wanted {target_per_rank:,} for step {step}).")
            process_chunk(buffer)
            buffer = []

            t = total
            print(f"\nStep {step}: {t:,} sequences total ({t - prev_total:,} new)")
            prev_total = t
            print("  Language distribution:")
            for lang in LANGS + ["other"]:
                if lang_seq_counts[lang] > 0:
                    pct = lang_seq_counts[lang] / t * 100
                    print(f"    {lang}: {lang_seq_counts[lang]:>10,} seqs ({pct:5.1f}%), "
                          f"{len(per_lang_counts.get(lang, {})):,} unique words")

            output = {
                "step": step,
                "total_sequences": t,
                "min_count_threshold": args.min_count,
                "lang_seq_counts": dict(lang_seq_counts),
                "per_lang": {},
            }
            for lang in LANGS:
                if per_lang_counts.get(lang):
                    filtered = {w: c for w, c in per_lang_counts[lang].items() if c >= args.min_count}
                    output["per_lang"][lang] = {
                        "total_unique_words": len(per_lang_counts[lang]),
                        "words_above_threshold": len(filtered),
                        "word_counts": filtered,
                    }
            all_filtered = {w: c for w, c in all_counts.items() if c >= args.min_count}
            output["per_lang"]["all"] = {
                "total_unique_words": len(all_counts),
                "words_above_threshold": len(all_filtered),
                "word_counts": all_filtered,
            }
            out_path = os.path.join(args.input_dir, f"frequent_words_step{step}.json")
            # encoding="utf-8" is REQUIRED: ensure_ascii=False writes raw vi/zh/ru/de/ar
            # chars, which crash with UnicodeEncodeError under a C/POSIX locale otherwise.
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"  Saved to {out_path}")
    finally:
        pool.close()
        pool.join()
        for f in rank_files.values():
            f.close()

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for step in sorted_steps:
        with open(os.path.join(args.input_dir, f"frequent_words_step{step}.json"),
                  encoding="utf-8") as f:
            data = json.load(f)
        print(f"\nStep {step}:")
        for lang in LANGS:
            if lang in data["per_lang"]:
                print(f"  {lang}: {data['per_lang'][lang]['words_above_threshold']:,} words (>= {args.min_count})")
        print(f"  all: {data['per_lang']['all']['words_above_threshold']:,} words (>= {args.min_count})")


if __name__ == "__main__":
    main()
