"""
Team NOVA - AFML Project
final_eval.py: the held-out TEST evaluation. Run ONCE, after all development is finished.

Course guideline ("Care with evaluation"): train on the train set, measure progress on the validation
set, and only at the very end evaluate on the test set, once.

Test sets
  * Pretraining-data test = the second half of the validation file, cut at a document
    boundary (Base data mix: 4.50M tokens; the earlier TinyStories model: about 2.35M tokens).
      - Pretrained models are evaluated at their FINAL checkpoint (ckpt_last.pt, the last step).
        The step budget was fixed in advance and validation loss was only monitored, never used to
        choose the reported checkpoint, so the validation data had no influence on what is tested.
  * Fine-tuning test  = conversations / instructions held out by finetune.py and never used for training or
    for choosing a checkpoint: smol-smoltalk (Base model: 2,000 conversations of its test split, separate from
    the 2,000 used for validation) or Alpaca (small model: 5%).
      - Fine-tuned models are evaluated at ckpt_best.pt (chosen on the VALIDATION part).

Every test token is scored exactly once (non-overlapping windows), so the numbers are deterministic.

Usage:
  python final_eval.py --data_dir DATA --runs_dir RUNS --out_dir RESULTS
"""

import argparse
from contextlib import nullcontext
import json
import math
import os

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from finetune import instruction_val_loss
from routing_analysis import load_model
from train import val_dev_end


def pretrain_test_tokens(data_dir, eot_id):
    """Second half of val.bin, starting right after the first <|endoftext|> past the midpoint."""
    path = os.path.join(data_dir, "val.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    start = val_dev_end(path, eot_id)
    return np.asarray(data[start:], dtype=np.int64), start, len(data)


@torch.no_grad()
def lm_test_loss(model, tokens, device, batch=32):
    """Mean next-token cross-entropy over ALL test tokens, non-overlapping windows of block_size."""
    model.eval()
    T = model.cfg.block_size
    n_win = (len(tokens) - 1) // T
    total, count = 0.0, 0
    for s in range(0, n_win, batch):
        idx = np.arange(s, min(s + batch, n_win)) * T
        w = torch.from_numpy(np.stack([tokens[i : i + T + 1] for i in idx])).to(device)
        logits = model(w[:, :-1])[0]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), w[:, 1:].reshape(-1), reduction="sum")
        total += loss.item()
        count += w[:, 1:].numel()
    return total / count, count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(a.data_dir, "meta.json")))
    st = meta["special_tokens"]
    eot_id, pad_id = st["<|endoftext|>"], st["<|pad|>"]
    test_tokens, start, n_val = pretrain_test_tokens(a.data_dir, eot_id)
    print(f"Pretraining-data test: tokens {start:,} to {n_val:,} of val.bin ({len(test_tokens):,} tokens)")

    # held-out fine-tuning test set: smol-smoltalk conversations (Base model) or Alpaca (small model)
    sft_test, sft_name = None, None
    files = os.listdir(a.data_dir)
    for prefix, suffix, name in (("smoltalk_tok_", ".pt", "smoltalk"), ("alpaca_tok_", "_split3.pt", "alpaca")):
        caches = sorted(f for f in files if f.startswith(prefix) and f.endswith(suffix))
        if caches:
            sft_test, sft_name = torch.load(os.path.join(a.data_dir, caches[0]), weights_only=False)["test"], name
            print(f"{name} test: {len(sft_test):,} held-out examples")
            break

    runs = [("dense", "ckpt_last.pt", "pretrained, final checkpoint"),
            ("moe", "ckpt_last.pt", "pretrained, final checkpoint"),
            ("moe_nobal", "ckpt_last.pt", "pretrained, final checkpoint"),
            ("moe_early", "ckpt_last.pt", "pretrained, final checkpoint (expert placement ablation)"),
            ("dense_sft", "ckpt_best.pt", "instruction-tuned, best on chat validation"),
            ("moe_sft", "ckpt_best.pt", "instruction-tuned, best on chat validation"),
            ("moe_dpo", "ckpt_last.pt", "chat model after DPO (final checkpoint)")]
    rows = []
    for run, ck, note in runs:
        path = os.path.join(a.runs_dir, run, ck)
        if not os.path.exists(path):
            print(f"skip {run}: {path} not found")
            continue
        model = load_model(path, a.device)
        ts_loss, n_tok = lm_test_loss(model, test_tokens, a.device)
        row = {"model": run, "checkpoint": ck, "note": note,
               "total_params_M": round(model.num_params() / 1e6, 2),
               "active_params_M": round(model.num_active_params() / 1e6, 2),
               "pretrain_test_loss": round(ts_loss, 4),
               "pretrain_test_perplexity": round(math.exp(ts_loss), 3),
               "pretrain_test_tokens": n_tok}
        if sft_test is not None and run.endswith(("_sft", "_dpo")):
            il = instruction_val_loss(model, sft_test, pad_id, a.device, nullcontext())
            row[f"{sft_name}_test_loss (response tokens)"] = round(il, 4)
            row[f"{sft_name}_test_perplexity"] = round(math.exp(il), 3)
        rows.append(row)
        print(row)
        del model
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out_dir, "final_test_results.csv"), index=False)
    print(f"Saved {a.out_dir}/final_test_results.csv")


if __name__ == "__main__":
    main()
