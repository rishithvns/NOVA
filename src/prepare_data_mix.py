"""
Team NOVA - AFML Project
Base model, step 1: data preparation on a MIX of datasets (course feedback: "try multiple datasets to make
full use of the additional parameters and generalise beyond just generating stories").

Sources (share of training tokens, all English):
  fineweb_edu  50%  FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT): web pages filtered for educational
                    quality; the best-known open pretraining set for small models (used by SmolLM2)
  c4           30%  C4 (allenai/c4, en): cleaned general web text (the dataset recommended by the TA)
  wikipedia    20%  English Wikipedia (wikimedia/wikipedia, 20231101.en): factual, encyclopedic text

What this script does:
  1. Downloads a few files of each source from the Hugging Face Hub (to local disk, not Drive).
  2. Holds out VALIDATION documents from each source (C4: its own validation file; FineWeb-Edu and
     Wikipedia have no validation split, so the first documents read are held out and never trained on).
  3. Trains our OWN byte-level BPE tokenizer (vocab 16,384) on a sample of all three sources.
  4. Writes train.bin (each source's documents, up to its token budget) and val.bin (validation documents
     of all sources SHUFFLED together, so both the development half and the test half contain every source),
     plus val_source.bin: for every validation token, which source it came from (for per-source analysis).
  5. Saves meta.json (token counts and shares per source) and runs sanity checks.

Training reads random 512-token windows from train.bin, so each source is seen in proportion to its
share of train.bin's tokens.

Run (Colab, no GPU needed):
    pip install -q huggingface_hub tokenizers pyarrow
    python prepare_data_mix.py --out_dir data
Quick check without downloading anything:
    python prepare_data_mix.py --out_dir /tmp/mix_test --synthetic
"""

import argparse
import gzip
import json
import os
import random
import time

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|instruction|>", "<|response|>"]   # ids 0..3, as before

# name: (Hub repo, file prefix, file format, fallback file names, share of training tokens)
SOURCES = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample/10BT/", "parquet",
                    [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)], 0.50),
    "c4": ("allenai/c4", "en/c4-train.", "jsonl.gz",
           [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(1024)], 0.30),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en/", "parquet", [], 0.20),
}
C4_VAL = ("allenai/c4", "en/c4-validation.", [f"en/c4-validation.{i:05d}-of-00008.json.gz" for i in range(8)])


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
def list_files(repo, prefix, suffixes, fallback, n):
    """First n files of a Hub dataset repo whose path starts with prefix. Falls back to known names."""
    try:
        from huggingface_hub import HfApi
        files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                       if f.startswith(prefix) and f.endswith(suffixes))
        if len(files) >= n:
            return files[:n]
        print(f"  listing {repo} returned {len(files)} files for {prefix!r}")
    except Exception as e:
        print(f"  could not list {repo} ({type(e).__name__})")
    if len(fallback) >= n:
        print("  using standard file names")
        return fallback[:n]
    raise SystemExit(f"Cannot find files for {repo}/{prefix}. Check the dataset page on huggingface.co.")


def download(repo, files, cache_dir):
    from huggingface_hub import hf_hub_download
    return [hf_hub_download(repo, f, repo_type="dataset", cache_dir=cache_dir) for f in files]


# ----------------------------------------------------------------------------
# Reading documents
# ----------------------------------------------------------------------------
def read_texts(paths, fmt, min_chars=50):
    """Yield document texts one by one (never loads a whole file into memory)."""
    for path in paths:
        if fmt == "parquet":
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=2_000):
                for text in batch.column(0).to_pylist():
                    text = (text or "").strip()
                    if len(text) >= min_chars:
                        yield text
        else:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    text = json.loads(line)["text"].strip()
                    if len(text) >= min_chars:
                        yield text


def synthetic_texts(name, seed):
    """Tiny fake documents per source, for testing the pipeline without downloads. Each document carries
    a unique id so tests can check that no validation document leaks into training."""
    rng = random.Random(seed)
    words = {"fineweb_edu": "photosynthesis energy cell student lesson learn plant light water explain".split(),
             "c4": "buy online shop great price best free delivery today offer product".split(),
             "wikipedia": "born city river war century king population located university history".split()}[name]
    i = 0
    while True:
        body = " ".join(rng.choice(words) for _ in range(rng.randint(30, 120)))
        yield f"{name} doc{i:07d}. {body.capitalize()}."
        i += 1


def batches(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


# ----------------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------------
def train_tokenizer(doc_iters, shares, vocab_size, n_docs):
    """Byte-level BPE on a sample of every source, in proportion to the source shares."""
    def mixed():
        counts = {k: int(n_docs * s) for k, s in shares.items()}
        its = {k: iter(v) for k, v in doc_iters.items()}
        while any(counts.values()):
            for k in list(its):
                if counts[k] > 0:
                    try:
                        yield next(its[k])
                    except StopIteration:
                        counts[k] = 0
                        continue
                    counts[k] -= 1
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  min_frequency=2, show_progress=True)
    tok.train_from_iterator(batches(mixed(), 5_000), trainer=trainer)
    return tok


# ----------------------------------------------------------------------------
# Writing token files
# ----------------------------------------------------------------------------
def take_docs(it, tok, eot_id, token_budget, batch_size=2_000):
    """Pull documents from `it` and tokenize them until token_budget tokens (incl. <|endoftext|>) are
    collected. Returns a list of uint16 arrays, one per document."""
    out, total = [], 0
    for texts in batches(it, batch_size):
        for enc in tok.encode_batch(texts):
            ids = np.asarray(enc.ids + [eot_id], dtype=np.uint16)
            out.append(ids)
            total += ids.size
            if total >= token_budget:
                return out, total
    return out, total


def write_train(f, it, tok, eot_id, token_budget, name, batch_size=2_000):
    """Stream documents of one source into the open train.bin until its token budget is reached."""
    total, n_docs, next_report = 0, 0, 50_000_000
    for texts in batches(it, batch_size):
        ids, done = [], False
        for enc in tok.encode_batch(texts):
            ids.extend(enc.ids)
            ids.append(eot_id)
            n_docs += 1
            if total + len(ids) >= token_budget:          # stop at the document that reaches the budget
                done = True
                break
        arr = np.asarray(ids, dtype=np.uint16)
        arr.tofile(f)
        total += arr.size
        if total >= next_report:
            print(f"    {name}: {n_docs:,} docs, {total / 1e6:.0f}M tokens", flush=True)
            next_report += 50_000_000
        if done:
            break
    if total < token_budget:
        print(f"  WARNING: {name} ran out of documents at {total / 1e6:.0f}M tokens "
              f"(budget {token_budget / 1e6:.0f}M); add files with --files_{name}")
    return total, n_docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data")
    p.add_argument("--cache_dir", default="/content/hf_cache", help="local disk, NOT Drive")
    p.add_argument("--vocab_size", type=int, default=16384)
    p.add_argument("--train_tokens", type=float, default=800e6,
                   help="total training tokens across sources (6,000 steps x 65,536 = 393M are used per run; "
                        "the rest allows longer runs without repeating documents)")
    p.add_argument("--val_tokens", type=float, default=9e6, help="validation tokens across sources (half = test)")
    p.add_argument("--tok_train_docs", type=int, default=300_000, help="documents used to learn the BPE merges")
    p.add_argument("--files_fineweb_edu", type=int, default=1, help="~2.15 GB, ~700M tokens each")
    p.add_argument("--files_c4", type=int, default=2, help="~356k documents, ~170M tokens each")
    p.add_argument("--files_wikipedia", type=int, default=3, help="~60M+ tokens each")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synthetic", action="store_true", help="fake documents, no download (pipeline test)")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    names = list(SOURCES)
    shares = {k: SOURCES[k][4] for k in names}

    # ---- document streams per source: one for validation, one for tokenizer training, one for train.bin
    if a.synthetic:
        # same stream as training (like FineWeb-Edu / Wikipedia): validation = the first documents, training skips them
        val_it = {k: synthetic_texts(k, a.seed + i) for i, k in enumerate(names)}
        def fresh(k, skip):                       # same stream as training documents, after the held-out ones
            it = synthetic_texts(k, a.seed + i_of[k])
            for _ in range(skip):
                next(it)
            return it
        i_of = {k: i for i, k in enumerate(names)}
    else:
        paths, fmts = {}, {}
        for k in names:
            repo, prefix, fmt, fallback, _ = SOURCES[k]
            n = getattr(a, f"files_{k}")
            suff = (".parquet",) if fmt == "parquet" else (".json.gz",)
            files = list_files(repo, prefix, suff, fallback, n)
            print(f"Downloading {k}: {files}")
            paths[k], fmts[k] = download(repo, files, a.cache_dir), fmt
        c4_val_files = list_files(C4_VAL[0], C4_VAL[1], (".json.gz",), C4_VAL[2], 1)
        c4_val_paths = download(C4_VAL[0], c4_val_files, a.cache_dir)
        def fresh(k, skip):
            it = read_texts(paths[k], fmts[k])
            for _ in range(skip):
                next(it)
            return it
        val_it = {k: (read_texts(c4_val_paths, "jsonl.gz") if k == "c4" else fresh(k, 0)) for k in names}

    # ---- tokenizer (trained on documents AFTER the held-out validation documents of each source)
    print(f"\nTraining BPE tokenizer (vocab {a.vocab_size}) on {a.tok_train_docs:,} documents from all sources ...")
    # hold out a fixed number of documents per source, generous enough for its share of the validation budget
    val_docs_text = {}
    for k in names:
        need = a.val_tokens * shares[k]
        n_hold = max(200, int(need / 150))                    # generous: documents average well over 150 tokens
        val_docs_text[k] = [next(val_it[k]) for _ in range(n_hold)]
    skip = {k: (0 if k == "c4" else len(val_docs_text[k])) for k in names}   # C4 val comes from its own file
    tok = train_tokenizer({k: fresh(k, skip[k]) for k in names}, shares, a.vocab_size, a.tok_train_docs)
    tok.save(os.path.join(a.out_dir, "tokenizer.json"))
    eot_id = tok.token_to_id("<|endoftext|>")

    # ---- validation: per source up to its share of val_tokens, then shuffle documents of all sources together
    print("\nTokenizing validation documents ...")
    val_docs, val_counts = [], {}
    for si, k in enumerate(names):
        docs, total = take_docs(iter(val_docs_text[k]), tok, eot_id, a.val_tokens * shares[k])
        val_docs += [(si, d) for d in docs]
        val_counts[k] = {"tokens": int(total), "documents": len(docs)}
        print(f"  {k}: {total:,} tokens, {len(docs):,} documents")
    random.Random(a.seed).shuffle(val_docs)
    val_ids = np.concatenate([d for _, d in val_docs])
    val_src = np.concatenate([np.full(d.size, si, dtype=np.uint8) for si, d in val_docs])
    val_ids.tofile(os.path.join(a.out_dir, "val.bin"))
    val_src.tofile(os.path.join(a.out_dir, "val_source.bin"))

    # ---- training: each source's documents (after its held-out validation documents) up to its budget
    print("\nTokenizing training documents (the slow step) ...")
    train_counts, start = {}, 0
    with open(os.path.join(a.out_dir, "train.bin"), "wb") as f:
        for k in names:
            n, d = write_train(f, fresh(k, skip[k]), tok, eot_id, a.train_tokens * shares[k], k)
            train_counts[k] = {"tokens": int(n), "documents": d, "range": [start, start + int(n)]}
            start += int(n)
            print(f"  {k}: {n:,} tokens ({d:,} documents)")
    n_train = sum(v["tokens"] for v in train_counts.values())

    meta = {
        "dataset": "mix: " + ", ".join(f"{k} {int(100 * shares[k])}%" for k in names),
        "sources": names, "target_shares": shares,
        "train_by_source": train_counts, "val_by_source": val_counts,
        "actual_train_shares": {k: round(v["tokens"] / n_train, 4) for k, v in train_counts.items()},
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": {t: tok.token_to_id(t) for t in SPECIAL_TOKENS},
        "train_tokens": n_train, "val_tokens": int(val_ids.size),
        "train_stories": sum(v["documents"] for v in train_counts.values()),   # "stories" = documents
        "val_stories": len(val_docs),
        "avg_tokens_per_story": round(n_train / max(1, sum(v["documents"] for v in train_counts.values())), 1),
        "dtype": "uint16", "val_source_file": "val_source.bin (uint8, index into 'sources')",
        "synthetic": bool(a.synthetic),
    }
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- sanity checks
    sample = "The quick brown fox jumps over the lazy dog. In 2019, researchers published a new study!"
    assert tok.decode(tok.encode(sample).ids) == sample, "round-trip failed"
    print(f"\n[check] round-trip OK: {len(sample)} chars -> {len(tok.encode(sample).ids)} tokens")
    assert val_src.size == val_ids.size and int(val_ids.max()) < tok.get_vocab_size()
    half = val_src.size // 2
    for part, s in (("first", val_src[:half]), ("second", val_src[half:])):
        shares_part = {k: round(float((s == i).mean()), 3) for i, k in enumerate(names)}
        print(f"[check] {part} half of validation, token share per source: {shares_part}")
    end = int(np.argmax(val_ids == eot_id))
    print(f"[check] first validation document ({names[val_src[0]]}, {end} tokens):")
    print("   ", tok.decode(val_ids[:end].tolist())[:300].replace("\n", " "), "...")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Metadata:\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
