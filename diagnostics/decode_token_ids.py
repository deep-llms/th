"""Step 1 of 2: decode training token IDs to text.

Reads ids_rank{N}.jsonl (produced by smoke_train.py --save_token_ids) and writes
text_rank{N}.jsonl with the SAME line ordering (line i of text = decode of line i
of ids). Single process, transformers only — NO multiprocessing, so there is no
transformers-vs-multiprocessing interaction. batch_decode uses the Rust
tokenizer's internal threads for parallelism.

Then run count_words_from_text.py (fasttext multiprocessing, no transformers) on
the text_rank*.jsonl files.

Usage:
  python diagnostics/decode_token_ids.py --input-dir /path/to/token_ids
"""

import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True,
                    help="Dir with ids_rank*.jsonl + meta.json")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write text_rank*.jsonl (default: same as input-dir)")
    ap.add_argument("--tokenizer", default=None,
                    help="Override tokenizer; default reads meta.json's tokenizer")
    ap.add_argument("--decode-batch", type=int, default=4000,
                    help="batch_decode sub-batch size")
    args = ap.parse_args()

    out_dir = args.output_dir or args.input_dir
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(args.input_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"{meta_path} not found (written by smoke_train.py --save_token_ids).")
    with open(meta_path) as f:
        meta = json.load(f)
    world_size = meta["world_size"]
    tok_name = args.tokenizer or meta.get("tokenizer")
    if tok_name is None:
        raise ValueError("No tokenizer in meta.json and --tokenizer not given.")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    print(f"Tokenizer: {tok_name}")
    print(f"Decoding {world_size} rank files from {args.input_dir} -> {out_dir}")

    def flush(buf, fout):
        # batch_decode preserves order; one output line per input id-list.
        for text in tokenizer.batch_decode(buf, skip_special_tokens=True):
            fout.write(json.dumps(text) + "\n")  # JSON-encoded string (ASCII-safe, escapes newlines)

    for rank in range(world_size):
        in_path = os.path.join(args.input_dir, f"ids_rank{rank}.jsonl")
        if not os.path.isfile(in_path):
            print(f"WARNING: missing {in_path}, skipping rank {rank}")
            continue
        out_path = os.path.join(out_dir, f"text_rank{rank}.jsonl")
        n = 0
        buf = []
        with open(in_path, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                try:
                    ids = json.loads(line)
                except json.JSONDecodeError:
                    # Preserve line alignment: a bad line becomes an empty sequence.
                    ids = []
                buf.append(ids)
                if len(buf) >= args.decode_batch:
                    flush(buf, fout)
                    n += len(buf)
                    buf = []
                    if n % 200000 < args.decode_batch:
                        print(f"  rank{rank}: {n:,} decoded")
            if buf:
                flush(buf, fout)
                n += len(buf)
        print(f"rank{rank}: {n:,} sequences -> {out_path}")

    # Ensure meta.json is available next to the text files for the counting step.
    if out_dir != args.input_dir:
        shutil.copy(meta_path, os.path.join(out_dir, "meta.json"))

    print("Done. Next: count_words_from_text.py --input-dir " + out_dir)


if __name__ == "__main__":
    main()
