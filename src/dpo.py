"""
Team NOVA - AFML Project
dpo.py: Direct Preference Optimization (DPO) of the chat model (extension of Phase 3).

Starts from the chat (SFT) checkpoint and trains it on preference pairs: for each prompt, a CHOSEN and a
REJECTED answer. DPO (Rafailov et al., 2023) needs no reward model: with the frozen SFT model as reference,

    loss = -log sigmoid( beta * [ (log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected)) ] )

where log pi(.) is the summed log-probability of the answer tokens given the prompt. The policy is pushed to
raise the chosen answer relative to the rejected one, while beta keeps it close to the reference model.

Data: Intel/orca_dpo_pairs (about 12.9k pairs; GPT-4 answers chosen over weaker answers). Each pair is
formatted as in fine-tuning: <|instruction|>prompt<|response|>answer<|endoftext|>. Pairs that do not fit in
the 512-token context are dropped. A fixed held-out set of pairs measures the preference accuracy.

Logged every eval: preference accuracy on held-out pairs (share where the policy prefers the chosen answer
more than the reference does), the share of held-out pairs where the model itself prefers the chosen answer,
the mean implicit reward margin, and the chat validation loss on smol-smoltalk (to check that DPO does not
undo the chat tuning).

Example:
    python dpo.py --data_dir DATA --init_ckpt RUNS/moe_sft/ckpt_best.pt --out_dir RUNS/moe_dpo
"""

import argparse
import copy
import csv
import glob
import json
import math
import os
import random
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import numpy as np

from model import ModelConfig, NovaLM


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--init_ckpt", required=True, help="the chat (SFT) checkpoint; also the frozen reference")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", default="Intel/orca_dpo_pairs")
    p.add_argument("--n_train", type=int, default=4000, help="training pairs")
    p.add_argument("--n_eval", type=int, default=500, help="held-out pairs for preference accuracy")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--min_lr", type=float, default=5e-7)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--batch_pairs", type=int, default=8, help="pairs per optimizer step")
    p.add_argument("--micro_pairs", type=int, default=4, help="pairs per forward pass (memory)")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--aux_coef", type=float, default=0.01)
    p.add_argument("--z_coef", type=float, default=1e-3)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def read_pairs(name):
    """Rows of the preference dataset as dicts. Uses `datasets` when it can be imported; otherwise reads the
    Hub's parquet/jsonl files directly (on Windows, Smart App Control can block the `datasets` library)."""
    try:
        from datasets import load_dataset
        return [dict(r) for r in load_dataset(name, split="train")]
    except (ImportError, OSError):
        pass
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    for rev in (None, "refs/convert/parquet"):
        try:
            files = sorted(api.list_repo_files(name, repo_type="dataset", revision=rev))
        except Exception:
            continue
        pick = [f for f in files if f.endswith(".parquet") and "test" not in f.split("/")[-1]]
        pick = pick or [f for f in files if f.endswith((".jsonl", ".json")) and not f.endswith("config.json")]
        if not pick:
            continue
        dfs = []
        for f in pick:
            path = hf_hub_download(name, f, repo_type="dataset", revision=rev)
            dfs.append(pd.read_parquet(path) if f.endswith(".parquet") else pd.read_json(path, lines=f.endswith(".jsonl")))
        return pd.concat(dfs, ignore_index=True).to_dict("records")
    raise RuntimeError(f"could not read {name}")


def encode_pair(tok, sp, row, block_size):
    """(prompt_ids, chosen_ids, rejected_ids) or None if the pair does not fit or the answers are identical."""
    system = (row.get("system") or "").strip()
    question = (row.get("question") or row.get("prompt") or "").strip()
    chosen, rejected = row.get("chosen"), row.get("rejected")
    if isinstance(chosen, (list, np.ndarray)):              # chat-format datasets: take the last assistant message
        chosen = chosen[-1]["content"]
        rejected = rejected[-1]["content"]
    chosen, rejected = (chosen or "").strip(), (rejected or "").strip()
    if not question or not chosen or not rejected or chosen == rejected:
        return None
    text = (system + "\n\n" + question) if system else question
    p = [sp["<|instruction|>"]] + tok.encode(text).ids + [sp["<|response|>"]]
    c = tok.encode(chosen).ids + [sp["<|endoftext|>"]]
    r = tok.encode(rejected).ids + [sp["<|endoftext|>"]]
    if len(p) + max(len(c), len(r)) > block_size + 1:
        return None
    return p, c, r


def load_pairs(args, tok, sp, block_size):
    cache = os.path.join(args.data_dir, f"dpo_{args.dataset.replace('/', '_')}_{block_size}_{args.n_train}_{args.n_eval}.pt")
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        print(f"Loaded cached preference pairs: {len(d['train'])} train / {len(d['eval'])} eval")
        return d
    rows = read_pairs(args.dataset)
    random.Random(args.seed).shuffle(rows)
    kept, dropped = [], 0
    for row in rows:
        e = encode_pair(tok, sp, row, block_size)
        if e is None:
            dropped += 1
            continue
        kept.append(e)
        if len(kept) >= args.n_train + args.n_eval:
            break
    d = {"eval": kept[:args.n_eval], "train": kept[args.n_eval:]}
    d["stats"] = {"dataset": args.dataset, "rows_read": len(rows), "kept": len(kept),
                  "dropped_too_long_or_invalid": dropped, "train": len(d["train"]), "eval": len(d["eval"]),
                  "mean_prompt_tokens": round(float(np.mean([len(p) for p, _, _ in kept])), 1),
                  "mean_chosen_tokens": round(float(np.mean([len(c) for _, c, _ in kept])), 1),
                  "mean_rejected_tokens": round(float(np.mean([len(r) for _, _, r in kept])), 1)}
    print("Preference data:", json.dumps(d["stats"]))
    torch.save(d, cache)
    return d


def make_pair_batch(pairs, pad_id, device):
    """Chosen and rejected sequences stacked: rows [0, B) chosen, [B, 2B) rejected.
    x = ids[:-1], y = ids[1:], m = 1 on answer tokens (the target is an answer token)."""
    seqs = [(p + c, len(p)) for p, c, _ in pairs] + [(p + r, len(p)) for p, _, r in pairs]
    L = max(len(s) for s, _ in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.float32)
    for b, (s, lp) in enumerate(seqs):
        ids[b, :len(s)] = torch.tensor(s)
        mask[b, lp:len(s)] = 1.0
    return ids[:, :-1].to(device), ids[:, 1:].to(device), mask[:, 1:].to(device)


def seq_logps(model, x, y, m, ctx):
    """Summed log-probability of the answer tokens of each sequence, plus the model's info dict."""
    with ctx:
        logits, _, info = model(x)
    logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (logp * m).sum(-1), info


def dpo_terms(pol, ref, beta):
    """pol/ref: summed log-probs, first half chosen, second half rejected."""
    B = pol.shape[0] // 2
    pc, pr, rc, rr = pol[:B], pol[B:], ref[:B], ref[B:]
    logits = beta * ((pc - rc) - (pr - rr))
    loss = -F.logsigmoid(logits).mean()
    return loss, logits.detach() / beta


@torch.no_grad()
def evaluate_pairs(model, ref, pairs, pad_id, device, ctx, beta, bs=8):
    """reward accuracy: share of pairs where the policy raised the chosen answer more than the rejected one
    (relative to the reference; 0 before training by definition); margin: mean implicit reward margin;
    prefers_chosen: share of pairs where the model itself gives the chosen answer a higher average
    per-token log-probability than the rejected one (meaningful before and after)."""
    model.eval()
    margins, prefer = [], []
    for s in range(0, len(pairs), bs):
        x, y, m = make_pair_batch(pairs[s:s + bs], pad_id, device)
        pol, _ = seq_logps(model, x, y, m, ctx)
        rl, _ = seq_logps(ref, x, y, m, ctx)
        _, margin = dpo_terms(pol, rl, beta)
        margins.append(margin.float().cpu())
        avg = (pol / m.sum(-1).clamp(min=1)).float().cpu()
        B = avg.shape[0] // 2
        prefer.append((avg[:B] > avg[B:]).float())
    model.train()
    mg = torch.cat(margins)
    return float((mg > 0).float().mean()), float(mg.mean()), float(torch.cat(prefer).mean())


@torch.no_grad()
def chat_val_loss(model, val, pad_id, device, ctx, bs=32):
    from finetune import instruction_val_loss
    return instruction_val_loss(model, val, pad_id, device, ctx, bs)


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.out_dir, "DONE.json")):
        print("DPO run already finished:", open(os.path.join(args.out_dir, "DONE.json")).read())
        return
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device
    from tokenizers import Tokenizer
    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    sp = meta["special_tokens"]
    pad_id = sp["<|pad|>"]
    tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))

    ck = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    c = dict(ck["config"])
    c["moe_layers"] = tuple(c["moe_layers"])
    cfg = ModelConfig(**c)
    model = NovaLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    ref = copy.deepcopy(model).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    is_moe = len(cfg.moe_layers) > 0

    data = load_pairs(args, tok, sp, cfg.block_size)
    train, held = data["train"], data["eval"]
    chat_val = None
    caches = sorted(glob.glob(os.path.join(args.data_dir, f"smoltalk_tok_{cfg.block_size}_*.pt")))
    if caches:
        chat_val = torch.load(caches[0], weights_only=False)["val"]

    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        ctx = nullcontext()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / args.batch_pairs)
    max_steps = max(1, int(args.epochs * steps_per_epoch))
    accum = max(1, args.batch_pairs // args.micro_pairs)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        prog = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(prog, 1.0)))

    log_path = os.path.join(args.out_dir, "log.csv")
    fields = ["step", "lr", "train_loss", "train_acc", "reward_acc", "reward_margin", "prefers_chosen", "chat_val_loss", "elapsed_min"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    def log_eval(step, tr_loss, tr_acc, t0):
        acc, mg, pc = evaluate_pairs(model, ref, held, pad_id, device, ctx, args.beta)
        cv = chat_val_loss(model, chat_val, pad_id, device, ctx) if chat_val else float("nan")
        row = [step, f"{lr_at(step):.2e}", f"{tr_loss:.4f}", f"{tr_acc:.3f}", f"{acc:.4f}", f"{mg:.4f}",
               f"{pc:.4f}", f"{cv:.4f}", f"{(time.time() - t0) / 60:.1f}"]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"step {step:5d} | lr {lr_at(step):.2e} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"held-out reward acc {acc:.3f} margin {mg:+.3f} prefers chosen {pc:.3f} | chat val loss {cv:.4f}", flush=True)
        return acc, mg, pc, cv

    print(f"DPO: {len(train)} train pairs, {len(held)} held-out pairs, {max_steps} steps x {args.batch_pairs} pairs, "
          f"beta {args.beta}, lr {args.lr}")
    t0 = time.time()
    acc0, mg0, pc0, cv0 = log_eval(0, float("nan"), float("nan"), t0)
    order = list(range(len(train)))
    rng.shuffle(order)
    pos, step = 0, 0
    run_loss, run_acc, n_run = 0.0, 0.0, 0
    model.train()
    while step < max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            if pos + args.micro_pairs > len(order):
                rng.shuffle(order)
                pos = 0
            batch = [train[i] for i in order[pos:pos + args.micro_pairs]]
            pos += args.micro_pairs
            x, y, m = make_pair_batch(batch, pad_id, device)
            pol, info = seq_logps(model, x, y, m, ctx)
            with torch.no_grad():
                rl, _ = seq_logps(ref, x, y, m, ctx)
            loss, margin = dpo_terms(pol, rl, args.beta)
            total = loss
            if is_moe and info["aux_loss"] is not None:
                total = total + args.aux_coef * info["aux_loss"] + args.z_coef * info["z_loss"]
            (total / accum).backward()
            run_loss += loss.item()
            run_acc += float((margin > 0).float().mean())
            n_run += 1
        if not all(torch.isfinite(p_.grad).all() for p_ in model.parameters() if p_.grad is not None):
            print(f"step {step}: non-finite gradient, skipped")
            step += 1
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1
        if step % args.eval_interval == 0 or step == max_steps:
            log_eval(step, run_loss / max(n_run, 1), run_acc / max(n_run, 1), t0)
            run_loss, run_acc, n_run = 0.0, 0.0, 0

    acc1, mg1, pc1 = evaluate_pairs(model, ref, held, pad_id, device, ctx, args.beta)
    cv1 = chat_val_loss(model, chat_val, pad_id, device, ctx) if chat_val else float("nan")
    torch.save({"model": model.state_dict(), "config": c, "step": step, "init_ckpt": args.init_ckpt,
                "dpo": vars(args)}, os.path.join(args.out_dir, "ckpt_last.pt.tmp"))
    os.replace(os.path.join(args.out_dir, "ckpt_last.pt.tmp"), os.path.join(args.out_dir, "ckpt_last.pt"))
    summary = {"steps": step, "pairs_train": len(train), "pairs_eval": len(held), "beta": args.beta,
               "reward_acc_after": round(acc1, 4), "reward_margin_after": round(mg1, 4),
               "prefers_chosen_before": round(pc0, 4), "prefers_chosen_after": round(pc1, 4),
               "chat_val_loss_before": round(cv0, 4), "chat_val_loss_after": round(cv1, 4),
               "minutes": round((time.time() - t0) / 60, 1), "data": data["stats"]}
    write_json(os.path.join(args.out_dir, "DONE.json"), summary)
    print("Done.", json.dumps(summary))


if __name__ == "__main__":
    main()
