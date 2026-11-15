"""
Team NOVA - AFML Project
bench_eval.py: standard zero-shot benchmarks for our SLMs (the evaluations language-model papers report).

Method (the same as EleutherAI's lm-evaluation-harness, multiple choice): for every answer choice, compute
the log-probability the model gives to the choice's tokens after the question (the context). The model's
answer is the choice with the highest score.
  acc       highest total log-probability
  acc_norm  highest log-probability per character of the choice (removes the bias toward short answers);
            the number usually reported for HellaSwag, ARC and PIQA
95% confidence interval: +- 1.96 x sqrt(acc (1 - acc) / n).

Tasks (all English, standard splits):
  hellaswag  Rowan/hellaswag, validation      pick the most sensible ending of an everyday situation (4 choices)
  arc_easy   allenai/ai2_arc ARC-Easy, test   grade-school science questions (3-5 choices)
  piqa       ybisk/piqa, validation           physical common sense: which of two solutions works (2 choices)
  sciq       allenai/sciq, test               science questions with a supporting paragraph (4 choices)
  arc_challenge  allenai/ai2_arc ARC-Challenge, test   the harder science questions (3-5 choices)
  openbookqa allenai/openbookqa, test         elementary science needing common knowledge (4 choices)
Random-guess accuracy: 25%, about 25%, 50%, 25%, about 25%, 25%.

Reference models scored with the SAME code (so the comparison is fair): public dense SLMs of similar size,
given as  ref:<Hugging Face repo>, e.g. ref:openai-community/gpt2 (124M), ref:HuggingFaceTB/SmolLM2-135M,
ref:EleutherAI/pythia-160m.

Usage:
  python bench_eval.py --data_dir DATA --out_dir RESULTS --limit 1000 \
      --ckpts RUNS/dense/ckpt_last.pt RUNS/moe/ckpt_last.pt --names dense moe
  (public models: --ckpts hf:ibm-granite/granite-3.1-1b-a400m-base --names granite_base)
"""

import argparse
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Task formats (lm-evaluation-harness conventions)
# ----------------------------------------------------------------------------
def _hs_clean(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def fmt_hellaswag(ex):
    ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    return _hs_clean(ex["activity_label"] + ": " + ctx), [" " + _hs_clean(e) for e in ex["endings"]], int(ex["label"])


def fmt_arc(ex):
    labels = list(ex["choices"]["label"])
    return ("Question: " + ex["question"] + "\nAnswer:", [" " + t for t in ex["choices"]["text"]],
            labels.index(ex["answerKey"]))


def fmt_piqa(ex):
    return "Question: " + ex["goal"] + "\nAnswer:", [" " + ex["sol1"], " " + ex["sol2"]], int(ex["label"])


def fmt_obqa(ex):
    labels = list(ex["choices"]["label"])
    return ex["question_stem"], [" " + t for t in ex["choices"]["text"]], labels.index(ex["answerKey"].strip())


def fmt_sciq(ex):
    ctx = (ex["support"].strip() + "\n" if ex.get("support") else "") + "Question: " + ex["question"] + "\nAnswer:"
    return ctx, [" " + ex["distractor1"], " " + ex["distractor2"], " " + ex["distractor3"], " " + ex["correct_answer"]], 3


TASKS = {
    "hellaswag": dict(path="Rowan/hellaswag", name=None, split="validation", fmt=fmt_hellaswag, chance=0.25),
    "arc_easy": dict(path="allenai/ai2_arc", name="ARC-Easy", split="test", fmt=fmt_arc, chance=0.25),
    "piqa": dict(path="ybisk/piqa", name=None, split="validation", fmt=fmt_piqa, chance=0.5),
    "sciq": dict(path="allenai/sciq", name=None, split="test", fmt=fmt_sciq, chance=0.25),
    "arc_challenge": dict(path="allenai/ai2_arc", name="ARC-Challenge", split="test", fmt=fmt_arc, chance=0.25),
    "openbookqa": dict(path="allenai/openbookqa", name="main", split="test", fmt=fmt_obqa, chance=0.25),
}


class RefLM(torch.nn.Module):
    """A public dense Hugging Face causal LM behind the interface choice_logprobs expects."""

    def __init__(self, repo, device):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(device).eval()
        n_pos = getattr(self.model.config, "max_position_embeddings", None) or 1024
        self.cfg = type("Cfg", (), {"block_size": min(1024, int(n_pos))})()

    def forward(self, x):
        return (self.model(input_ids=x).logits,)

    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())


def _hub_parquet_rows(t):
    """Read a benchmark split straight from the Hub's parquet files (no `datasets` library).
    Used when `datasets` cannot be imported (e.g. Windows Smart App Control blocks its xxhash DLL)."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")       # plain HTTP download (no extra binary helper)
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    for rev in (None, "refs/convert/parquet"):
        try:
            files = api.list_repo_files(t["path"], repo_type="dataset", revision=rev)
        except Exception:
            continue
        pick = []
        for f in sorted(files):
            parts = f.split("/")
            if not f.endswith(".parquet"):
                continue
            if t["name"] and t["name"] not in parts[:-1]:
                continue
            if t["split"] in parts[:-1] or parts[-1].startswith(t["split"] + "-") or parts[-1].startswith(t["split"] + "."):
                pick.append(f)
        if pick:
            dfs = [pd.read_parquet(hf_hub_download(t["path"], f, repo_type="dataset", revision=rev)) for f in pick]
            return pd.concat(dfs, ignore_index=True).to_dict("records")
    raise RuntimeError(f"no parquet files found for {t['path']} {t['name']} {t['split']}")


def load_task(task, limit=None, seed=0):
    t = TASKS[task]
    try:
        from datasets import load_dataset
        args = (t["path"],) + ((t["name"],) if t["name"] else ())
        try:
            ds = load_dataset(*args, split=t["split"])
        except Exception:                                   # script-based datasets: use the Hub's parquet copy
            ds = load_dataset(*args, split=t["split"], revision="refs/convert/parquet")
        rows = ds
    except (ImportError, OSError):                          # `datasets` blocked on this machine: read parquet directly
        rows = _hub_parquet_rows(t)
    if limit and len(rows) > limit:
        # fixed random subset, same for every model; identical to datasets' shuffle(seed).select(range(limit))
        idx = np.random.default_rng(seed).permutation(len(rows))[:limit]
        rows = [rows[int(i)] for i in idx]
    return [t["fmt"](ex) for ex in rows]


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def make_encoder(tok):
    """Our tokenizers.Tokenizer or a Hugging Face tokenizer -> function text -> list of ids."""
    if hasattr(tok, "encode") and not hasattr(tok, "convert_tokens_to_ids"):
        return lambda s: tok.encode(s).ids
    return lambda s: tok(s, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def choice_logprobs(model, encode, ctx, choices, device, pad_id=0):
    """Sum of log-probabilities of each choice's tokens given the context. Long inputs keep their END
    (the context is cut from the left) so they fit in the model's window."""
    T = model.cfg.block_size
    c_ids = encode(ctx)
    seqs, spans = [], []
    for ch in choices:
        a = encode(ch)
        full = (c_ids + a)[-(T + 1):]
        n_a = min(len(a), len(full) - 1)
        seqs.append(full)
        spans.append(n_a)
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for i, sq in enumerate(seqs):
        x[i, : len(sq)] = torch.tensor(sq)
    x = x.to(device)
    logits = model(x[:, :-1])[0].float()
    logp = F.log_softmax(logits, dim=-1)
    out = []
    for i, (sq, n_a) in enumerate(zip(seqs, spans)):
        tgt = torch.tensor(sq[len(sq) - n_a:], device=device)
        pos = torch.arange(len(sq) - n_a - 1, len(sq) - 1, device=device)
        out.append(float(logp[i, pos, tgt].sum()))
    return out


def evaluate_task(model, encode, examples, device):
    acc = acc_n = 0
    for ctx, choices, gold in examples:
        lp = choice_logprobs(model, encode, ctx, choices, device)
        acc += int(int(np.argmax(lp)) == gold)
        acc_n += int(int(np.argmax([l / max(1, len(c)) for l, c in zip(lp, choices)])) == gold)
    n = len(examples)
    se = lambda p: 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))
    return {"n": n, "acc": round(acc / n, 4), "acc_ci": round(se(acc / n), 4),
            "acc_norm": round(acc_n / n, 4), "acc_norm_ci": round(se(acc_n / n), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="our data folder (tokenizer.json) for our checkpoints")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--limit", type=int, default=1000, help="examples per task (fixed random subset)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    import pandas as pd
    from tokenizers import Tokenizer
    from routing_analysis import load_model
    os.makedirs(a.out_dir, exist_ok=True)
    data = {}
    for t in a.tasks:
        try:
            data[t] = load_task(t, a.limit)
            print(f"{t}: {len(data[t])} examples")
        except Exception as e:
            print(f"skip {t}: could not load ({type(e).__name__}: {e})")
    rows = [{"model": "random guess", "task": t, "n": len(ex), "acc": TASKS[t]["chance"],
             "acc_norm": TASKS[t]["chance"]} for t, ex in data.items()]
    for ck, name in zip(a.ckpts, a.names):
        if str(ck).startswith("ref:"):                  # public dense reference model
            from transformers import AutoTokenizer
            model, tok = RefLM(ck[4:], a.device), AutoTokenizer.from_pretrained(ck[4:])
            enc = make_encoder(tok)
            for t, ex in data.items():
                r = evaluate_task(model, enc, ex, a.device)
                rows.append({"model": name, "task": t, **r})
                print(name, t, r)
            del model
            continue
        model = load_model(ck, a.device)
        if str(ck).startswith("hf:"):
            from transformers import AutoTokenizer
            from hf_moe import parse_spec
            tok = AutoTokenizer.from_pretrained(parse_spec(ck)[0])
        else:
            tok = Tokenizer.from_file(os.path.join(a.data_dir, "tokenizer.json"))
        enc = make_encoder(tok)
        for t, ex in data.items():
            r = evaluate_task(model, enc, ex, a.device)
            rows.append({"model": name, "task": t, **r})
            print(name, t, r)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, "benchmarks.csv"), index=False)
    print(df.pivot(index="model", columns="task", values="acc_norm"))


if __name__ == "__main__":
    main()
